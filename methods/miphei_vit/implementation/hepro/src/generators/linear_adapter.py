'''
vit + linear probing
'''
"""
MIPHEI-vit model inspired by ViTMatte model.
This is a modified version of the original code from the repository:
https://github.com/hustvl/ViTMatte/blob/main/modeling/meta_arch/vitmatte.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.layers import resample_abs_pos_embed
from timm.models import VisionTransformer, SwinTransformer

from .unet import SegmentationHead, initialize_decoder_head
from .lora import apply_lora
from .foundation_models import FOUNDATION_MODEL_REGISTRY


class ViTMatte(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):

        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs

    # def initialize(self):
    #     initialize_decoder_head(self.decoder)

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
        features = self.vit(x)  # b, l, d
        # print(features.shape, self.num_prefix_tokens)
        if self.is_swint:
            features = features.permute(0, 3, 1, 2)
        else:
            features = features[:, self.num_prefix_tokens:]
            features = features.permute(0, 2, 1)
            features = features.view((-1, self.embed_dim, *self.grid_size)) # b, d, h/p, w/p

        if self.scale_factor is not None:
            features = F.interpolate(features, scale_factor=self.scale_factor, mode="bicubic")
        return features


class LinearProbeHead(nn.Module):
    def __init__(self, emb_chans,  out_chans, target_size, activation=nn.Tanh()):
        super().__init__()

        self.target_size = target_size
        self.num_heads = out_chans
        for i in range(self.num_heads):
            setattr(self, f'linear_layer_{i}', nn.Conv2d(emb_chans, 1, kernel_size=1))

        self.activation = activation


    def forward(self, x):

        outputs = []
        for idx_head in range(self.num_heads):
            linear_head = getattr(self, f'linear_layer_{idx_head}')
            logits = linear_head(x)
            upsampled_logits = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
            output = self.activation(upsampled_logits)

            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)

        return outputs

class LinearProbeHeadv2(nn.Module):
    # single head
    def __init__(self, emb_chans,  out_chans, target_size, activation=nn.Tanh()):
        super().__init__()

        self.target_size = target_size
        self.num_heads = out_chans
        self.linear_layer = nn.Conv2d(emb_chans, out_chans, kernel_size=1)

        self.activation = activation


    def forward(self, x):

        logits = self.linear_layer(x)
        upsampled_logits = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        outputs = self.activation(upsampled_logits)

        return outputs



def get_vit_linear_proj(encoder_name, img_size, num_classes, use_lora=False, ckpt_path=None, drop_path_rate=0):
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")
    if use_lora:
        apply_lora(vit, rank=8, alpha=1.)
    else:
        vit.eval()
        for param in vit.parameters(): param.requires_grad = False
    encoder = Encoder(vit)
    decoder = LinearProbeHead(emb_chans=encoder.embed_dim, out_chans=num_classes, target_size=vit.patch_embed.img_size,
                             activation=nn.Tanh())
    model = ViTMatte(encoder=encoder, decoder=decoder)
    return model

def get_vit_linear_proj_v2(encoder_name, img_size, num_classes, use_lora=False, frozen=True, ckpt_path=None, drop_path_rate=0):
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")
    if use_lora:
        apply_lora(vit, rank=8, alpha=1.)
    else:
        vit.eval()
    encoder = Encoder(vit)
    decoder = LinearProbeHeadv2(emb_chans=encoder.embed_dim, out_chans=num_classes, target_size=vit.patch_embed.img_size,
                             activation=nn.Tanh())
    model = ViTMatte(encoder=encoder, decoder=decoder)
    if not use_lora and frozen:
        for param in model.encoder.vit.parameters():
            param.requires_grad = False
        model.encoder.vit.eval()

    return model