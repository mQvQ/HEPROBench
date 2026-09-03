# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .foundation_models import FOUNDATION_MODEL_REGISTRY
from timm.layers import LayerNorm2d
from .lora import apply_lora


class Encoder(nn.Module):
    def __init__(
        self,
        pretrained_vit,
    ):
        super().__init__()

        # self.backbone = timm.create_model(
        #     backbone_name,
        #     pretrained=ckpt_path is None,
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     num_classes=0,
        # )
        self.backbone = pretrained_vit

        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        img_size,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.img_size = img_size

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        # for i in range(num_classes):
        #     setattr(self, f'linear_layer_{i}', nn.Linear(self.encoder.backbone.embed_dim, 1))

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        print('patch size', patch_size)
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        print(num_upscale, self.encoder.backbone.embed_dim)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)


        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )
        # print(x.shape)
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(self, module: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor]):
        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        x = self.encoder.backbone.patch_embed(x)
        x = self.encoder.backbone._pos_embed(x)
        x = self.encoder.backbone.patch_drop(x)
        x = self.encoder.backbone.norm_pre(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer, preds_per_layer = [], [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                interpolated = F.interpolate(
                    mask_logits,
                    self.encoder.backbone.patch_embed.grid_size,
                    mode="bilinear",
                )
                interpolated = interpolated.view(
                    interpolated.size(0), interpolated.size(1), -1
                )
                attn_mask[
                    :,
                    : self.num_q,
                    self.num_q + self.encoder.backbone.num_prefix_tokens :,
                ] = (
                    interpolated > 0
                )
                attn_mask = self._disable_attn_mask(
                    attn_mask,
                    self.attn_mask_probs[
                        i - len(self.encoder.backbone.blocks) + self.num_blocks
                    ],
                )

            x = x + block.drop_path1(
                block.ls1(self._attn(block.attn, block.norm1(x), attn_mask))
            )
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        # calc preds
        mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
        preds = to_per_pixel_logits_semantic(mask_logits, class_logits)
        preds_per_layer.append(preds)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return preds_per_layer[-1]

def to_per_pixel_logits_semantic(mask_logits: torch.Tensor, class_logits: torch.Tensor):

    # preds = torch.einsum("bqhw, bqc -> bchw", mask_logits.sigmoid(), class_logits.softmax(dim=-1)[..., :-1],)
    class_logits = torch.tanh(class_logits) # regression value
    mask_logits = torch.softmax(mask_logits, dim=1) # q： weighed sum=1
    preds = torch.einsum("bqhw, bqc -> bchw", mask_logits, class_logits)

    return preds


def get_eomt(encoder_name, img_size, num_classes, use_lora=False, frozen=True, ckpt_path=None, drop_path_rate=0):
    vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
        img_size, ckpt_path=ckpt_path, drop_path_rate=drop_path_rate, global_pool="")
    if use_lora:
        apply_lora(vit, rank=8, alpha=1.)
    encoder = Encoder(vit)

    # decoder = Detail_Capture(emb_chans=encoder.embed_dim, out_chans=num_classes, use_attention=True,
    #                          activation=nn.Tanh())
    # model = ViTMatte(encoder=encoder, decoder=decoder)
    model = EoMT(encoder, img_size, num_classes=num_classes, num_q=100)
    if not use_lora and frozen:
        for param in model.backbone.vit.parameters():
            param.requires_grad = False
        model.encoder.backbone.eval()

    return model

if __name__ == "__main__":
    pass


    # encoder_name = 'pathgen'
    # img_size = 256
    # vit = FOUNDATION_MODEL_REGISTRY[encoder_name](
    #     img_size, ckpt_path=None, drop_path_rate=0, global_pool="")
    #
    # encoder = ViT(vit)
    #
    # model = EoMT(encoder, num_classes=7, num_q=100)
    # input = torch.randn(2, 3, 256, 256)
    # mask_logit_prob, class_logit_prob = model(input)
    # for i in range(len(mask_logit_prob)):
    #     mask_logits = F.interpolate(mask_logit_prob[i], img_size, mode="bilinear")
    #     preds = to_per_pixel_logits_semantic(mask_logits, class_logit_prob[i])
    #     print(preds.shape)