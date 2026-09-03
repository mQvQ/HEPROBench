import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from .dpt_blocks import FeatureFusionBlock, _make_scratch
from .foundation_models import FOUNDATION_MODEL_REGISTRY
from .lora import apply_lora

class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        stride=2,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    def __init__(
        self,
        in_chans = 4,
        out_chans = [48, 96, 192],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)
        
        for i in range(len(self.conv_chans)-1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i+1]
            self.convs.append(
                Basic_Conv3x3(in_chan_, out_chan_)
            )
    
    def forward(self, x):
        out_dict = {'D0': x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            name_ = 'D'+str(i+1)
            out_dict[name_] = x
        
        return out_dict


class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """
    def __init__(
        self,
        in_chans,
        out_chans,
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) ## Nearest ?
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out


class ViTMatte(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.initialize()

    def forward(self, x):

        features = self.encoder(x)
        outputs = self.decoder(features, x)
        return outputs

    def initialize(self):
        initialize_decoder_head(self.decoder)

    def set_input_size(self, img_size):
        if any((s & (s - 1)) != 0 or s == 0 for s in img_size):
            raise ValueError("Both height and width in img_size must be powers of 2")
        if any(s < 128 for s in img_size):
            raise ValueError("Height and width must be greater or equal to 128")
        self.encoder.vit.set_input_size(img_size=img_size)
        self.encoder.grid_size = self.encoder.vit.patch_embed.grid_size


class Encoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        if not isinstance(vit, (VisionTransformer, SwinTransformer)):
            raise ValueError(f"Model should be a VisionTransformer or SwinTransformer, got {type(vit)}")
        self.vit = vit

        self.is_swint = isinstance(vit, SwinTransformer)
        self.grid_size = self.vit.patch_embed.grid_size
        if self.is_swint:
            self.num_prefix_tokens = 0
            self.embed_dim = self.vit.embed_dim * 2 ** (self.vit.num_layers -1)
        else:
            self.num_prefix_tokens = self.vit.num_prefix_tokens
            self.embed_dim = self.vit.embed_dim
        patch_size = self.vit.patch_embed.patch_size
        img_size = self.vit.patch_embed.img_size
        assert img_size[0] % 16 == 0
        assert img_size[1] % 16 == 0

        if self.is_swint:
            self.scale_factor = (2., 2.)
        else:
            if patch_size != (16, 16):
                target_grid_size = (img_size[0] / 16, img_size[1] / 16)
                self.scale_factor = (target_grid_size[0] / self.grid_size[0], target_grid_size[1] / self.grid_size[1])
            else:
                self.scale_factor = None

    def forward(self, x):
        features = self.vit(x)
        if self.is_swint:
            features = features.permute(0, 3, 1, 2)
        else:
            features = features[:, self.num_prefix_tokens:]
            features = features.permute(0, 2, 1)
            features = features.view((-1, self.embed_dim, *self.grid_size))
        if self.scale_factor is not None:
            features = F.interpolate(features, scale_factor=self.scale_factor, mode="bicubic")
        return features


class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """
    def __init__(
        self,
        emb_chans,
        in_chans=3,
        out_chans=1,
        convstream_out = [48, 96, 192],
        fusion_out = [256, 128, 64, 32],
        use_attention=True,
        activation=torch.nn.Identity()
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans=in_chans)
        self.conv_chans = self.convstream.conv_chans
        self.num_heads = out_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, emb_chans)
        for i in range(len(self.fus_channs)-1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans = self.fus_channs[i] + self.conv_chans[-(i+1)],
                    out_chans = self.fus_channs[i+1],
                )
            )

        for idx in range(self.num_heads):
            setattr(self, f'segmentation_head_{idx}', SegmentationHead(
                in_channels=fusion_out[-1],
                out_channels=1,
                activation=activation,
                kernel_size=3,
                use_attention=use_attention
            ))

    def forward(self, features, images):
        detail_features = self.convstream(images)
        for i in range(len(self.fusion_blks)):
            d_name_ = 'D'+str(len(self.fusion_blks)-i-1)
            features = self.fusion_blks[i](features, detail_features[d_name_])
        
        outputs = []
        for idx_head in range(self.num_heads):
            segmentation_head = getattr(self, f'segmentation_head_{idx_head}')
            output = segmentation_head(features)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)

        return outputs

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        if self.nclass > 1:
            out = self.scratch.output_conv(path_1)
        else:
            out = self.scratch.output_conv1(path_1)
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
            out = self.scratch.output_conv2(out)
        
        return out
        

class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        

        return depth.squeeze(1)


class HEPRO_HOPTIMUS(nn.Module):
    def __init__(self, encoder, num_classes, features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(HEPRO_HOPTIMUS, self).__init__()
        # "vit_giant_patch14_reg4_dinov2"
        self.vit = encoder
        self.grid_size = self.vit.patch_embed.grid_size
        dim = self.vit.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(num_classes, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        # features = self.vit.get_intermediate_layers(x, 4, return_class_token=True)
        features = self.vit.get_intermediate_layers(x, 4)
        # features= self.vit.get_intermediate_layers(x, 2)
        # print(len(features), features[-1].shape) [16, 324, 1536]
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)

class HEPRO_CTransPath(nn.Module):
    def __init__(self, encoder, num_classes, features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(HEPRO_CTransPath, self).__init__()
        # "vit_giant_patch14_reg4_dinov2"
        self.vit = encoder
        self.grid_size = self.vit.patch_embed.grid_size
        dim = self.vit.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(num_classes, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        # features = self.vit.get_intermediate_layers(x, 4, return_class_token=True)
        features = self.vit.get_intermediate_layers(x, 4)
        # features= self.vit.get_intermediate_layers(x, 2)
        # print(len(features), features[-1].shape) [16, 324, 1536]
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)

class HEPRO_Phikonv2(nn.Module):
    def __init__(self, encoder, num_classes, features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(HEPRO_Phikonv2, self).__init__()
        # "vit_giant_patch14_reg4_dinov2"
        self.vit = encoder
        self.grid_size = self.vit.patch_embed.grid_size
        dim = self.vit.blocks[0].attn.qkv.in_features
        # dim = self.vit.blocks[0].attn.qkv.dim
        self.depth_head = HEPRO_DPTHead(num_classes, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        # features = self.vit.get_intermediate_layers(x, 4, return_class_token=True)
        features = self.vit.get_intermediate_layers(x, 4)
        # features= self.vit.get_intermediate_layers(x, 2)
        # print(len(features), features[-1].shape) [16, 324, 1536]
        patch_h, patch_w = h // 16, w // 16
        depth = self.depth_head(features, patch_h, patch_w)
        
        
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        # depth = F.relu(depth)
        depth = torch.nn.Tanh()(depth)
        return depth.squeeze(1)

def get_hepro_hoptimus(encoder_name, img_size, num_classes, use_lora=False, ckpt_path=None, drop_path_rate=0):
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")
    
    if use_lora:
        apply_lora(vit, rank=8, alpha=1.)
    else:
        vit.eval()
    model = HEPRO_HOPTIMUS(encoder=vit, num_classes=num_classes, out_channels=[256, 512])
    
    return model


def get_hepro_phikonv2(encoder_name, img_size, num_classes, use_lora=False, ckpt_path=None, drop_path_rate=0):
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")
    
    # if use_lora:
    #     apply_lora(vit, rank=8, alpha=1.)
    # else:
    #     vit.eval()
    model = HEPRO_Phikonv2(encoder=vit, num_classes=num_classes)

    return model




class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    