import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from .dpt_blocks import FeatureFusionBlock, _make_scratch
from .foundation_models import FOUNDATION_MODEL_REGISTRY
from .lora import apply_lora
from timm.models import VisionTransformer, SwinTransformer

# 这是一个更优雅的封装方式，推荐使用
class SwinTFeatureExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers  # self.layers 存储了我们期望的顺序
        self._features = {layer: torch.empty(0) for layer in layers}
        self.hooks = []

        for layer_id in self.layers:
            layer = dict(self.model.named_modules())[layer_id]
            handle = layer.register_forward_hook(self.save_outputs_hook(layer_id))
            self.hooks.append(handle)

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    # --- 这里是唯一的修改 ---
    def __call__(self, x: torch.Tensor):
        # 模型正常执行前向传播
        _ = self.model(x)
        # 按照 self.layers 中定义的顺序，从字典中提取特征并返回一个列表
        return [self._features[layer_id] for layer_id in self.layers]

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()

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
            self.embed_dim = self.vit.embed_dim * 2 ** (self.vit.num_layers - 1)
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




def _make_fusion_block(features, use_bn, size=None):
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
    def __init__(self, nclass, in_channels, img_size, features=256, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], use_clstoken=False, is_swin=False):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.img_size = img_size
        self.is_swin = is_swin

        # Adapt in_channels for ViT (int) or Swin (list)
        if isinstance(in_channels, int):
            in_channels_list = [in_channels] * len(out_channels)
        else:
            if len(in_channels) != len(out_channels):
                raise ValueError("Length of in_channels list must match length of out_channels list.")
            in_channels_list = in_channels

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for in_channel, out_channel in zip(in_channels_list, out_channels)
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0], out_channels=out_channels[0],
                kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1], out_channels=out_channels[1],
                kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3], out_channels=out_channels[3],
                kernel_size=3, stride=2, padding=1)
        ])

        if use_clstoken:
            # This part is ViT-specific and might not be used with Swin
            self.readout_projects = nn.ModuleList()
            for in_channel in in_channels_list:
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channel, in_channel),
                        nn.GELU()))

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, self.nclass, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, out_features, grid_h=None, grid_w=None):
        if not self.is_swin and (grid_h is None or grid_w is None):
            raise ValueError("grid_h and grid_w must be provided for ViT features.")

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                # This logic is typically for ViT with a CLS token
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))

            # --- Reshaping logic based on backbone type ---
            if self.is_swin:
                # # For Swin: B, H*W, C -> B, C, H, W
                # b, n, c = x.shape
                # h = w = int(n ** 0.5)
                # x = x.permute(0, 2, 1).reshape(b, c, h, w)
                x = x.permute(0, 3, 1, 2)
            else:
                # For ViT: B, N, C -> B, C, H, W
                # Note: Here N is the number of patches, not H*W
                x = x.permute(0, 2, 1).reshape((x.shape[0], -1, grid_h, grid_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # print(layer_1_rn.shape, layer_2_rn.shape, layer_3_rn.shape, layer_4_rn.shape)
        # torch.Size([16, 96, 256, 256]) torch.Size([16, 96, 64, 64]) torch.Size([16, 96, 16, 16]) torch.Size([16, 96, 4, 4])
        path_4 = self.scratch.refinenet4(layer_4_rn)  # DPT's fusion blocks don't need size for Swin's feature map sizes
        # print(path_4.shape)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        out = self.scratch.output_conv2(out)

        return out
'''
class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, img_size, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024],
                 use_clstoken=False):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.img_size = img_size

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
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1,
                                              padding=1)

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, self.nclass, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, out_features, grid_h, grid_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x
            x = x.permute(0, 2, 1).reshape((x.shape[0], -1, grid_h, grid_w))

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

        out = self.scratch.output_conv1(path_1)
        # out = F.interpolate(out, (self.img_size, self.img_size), mode="bilinear", align_corners=True)
        out = F.interpolate(out, (self.img_size, self.img_size), mode="bilinear", align_corners=False)
        out = self.scratch.output_conv2(out)

        return out
'''
class DPT(nn.Module):

    def __init__(self, encoder, img_size, num_classes, features=256, out_channels = [256, 512, 1024, 1024], lora=False, use_bn=False, use_clstoken=False, localhub = True):
        super(DPT, self).__init__()
        self.encoder = encoder
        self.grid_size = self.encoder.patch_embed.grid_size
        self.img_size = img_size
        self.lora = lora
        self.is_swint = isinstance(encoder, SwinTransformer)
        if self.is_swint:
            # dim = self.encoder.layers[0].blocks[0].attn.qkv.in_features
            self.feature_extractor = SwinTFeatureExtractor(self.encoder,['layers.0', 'layers.1', 'layers.2', 'layers.3'])
            out_channels = [96, 192, 384, 768]
            dim = out_channels
            features = 96
        else:
            if self.lora:
                dim = self.encoder.blocks[0].attn.qkv.qkv.in_features
            else:
                dim = self.encoder.blocks[0].attn.qkv.in_features

        self.decoder = DPTHead(num_classes, dim, img_size, features, use_bn, out_channels=out_channels,
                                  use_clstoken=use_clstoken, is_swin=self.is_swint)

    def forward(self, x):

        h, w = x.shape[-2:]
        if self.is_swint:
            features = self.feature_extractor(x)
        else:
            features = self.encoder.get_intermediate_layers(x, 4)
        # print(len(features), features[-1].shape) # [16, 324, 1536]
        grid_h, grid_w = self.encoder.patch_embed.grid_size
        pred = self.decoder(features, grid_h, grid_w)

        return pred


def get_dpt(encoder_name, img_size, num_classes, use_lora=False, frozen=True, ckpt_path=None, drop_path_rate=0):
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")
    if use_lora:
        apply_lora(vit, rank=8, alpha=1.)
        model = DPT(encoder=vit, img_size=img_size, num_classes=num_classes, lora=True)
    else:
        model = DPT(encoder=vit, img_size=img_size, num_classes=num_classes)
    if not use_lora and frozen: # lora 一定不是frozen
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
    return model

if __name__ ==  "__main__":
    encoder_name = 'pathgen'
    img_size = 256
    model = get_dpt(encoder_name, img_size, 7)
    input = torch.randn(2, 3, 256, 256)
    preds = model(input)
    print(preds.shape)
