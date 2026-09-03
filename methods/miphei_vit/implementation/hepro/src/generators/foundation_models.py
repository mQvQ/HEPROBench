import os

import torch
import torch.nn as nn

import timm
from timm.layers import resample_abs_pos_embed
from timm.layers.helpers import to_2tuple
from timm.models import VisionTransformer, ResNet
from timm.models import load_state_dict_from_hf, parse_model_name
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import open_clip

'''
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L1246
'''
FOUNDATION_HF_CKPT_REGISTRY = {
    "univ2": "hf-hub:MahmoodLab/UNI2-h",
    "uni": "hf-hub:MahmoodLab/uni",
    "hoptimus0": "hf-hub:bioptimus/H-optimus-0",
    "provgigapath": "hf_hub:prov-gigapath/prov-gigapath",
    "sp85m": "hf_hub:MountSinaiCompPath/SP85M",
    "phikonv2": "hf_hub:owkin/phikon-v2",
    "restnet50_lunit_swav": "hf_hub:1aurent/resnet50.lunit_swav",
    "ctranspath": "hf_hub:jamesdolezal/CTransPath",
    "h0-mini": "hf-hub:bioptimus/H0-mini",
    "conch": "hf_hub:MahmoodLab/conch",
    "musk": "hf_hub:xiangjx/musk",
    "virchow2":"hf-hub:paige-ai/Virchow2",

}

def omiclip(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    model = timm.create_model(
        'vit_large_patch14_224', img_size=img_size, pre_norm=True, pretrained=False, num_classes=0, global_pool=global_pool,
        final_norm=False)

    if pretrained or ckpt_path:
        if pretrained:
            cached_file = hf_hub_download(
                repo_id="WangGuangyuLab/Loki",
                filename="checkpoint.pt",
                token=os.environ.get("HF_TOKEN")
            )
            state_dict = torch.load(cached_file, map_location='cpu', weights_only=False)['state_dict']
            state_dict = _convert_omiclip(state_dict, model, prefix='visual.')
            state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
            model.load_state_dict(state_dict, strict=False)

        else:
            print("Warning: random initialization for omiclip foundation model")
    else:
        print("Warning: random initialization for omiclip foundation model")

    return model


    # clip_model, preprocess = create_model_from_pretrained('coca_ViT-L-14', pretrained=cached_file, weights_only=False)


def musk(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    model = timm.create_model('beit3_large_patch16_224', img_size=img_size, dynamic_img_size=False, num_classes=0,
                              global_pool=global_pool, pre_norm=False, final_norm=True,
                              )  # 不确定
    _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["musk"])
    if pretrained or ckpt_path:
        if pretrained:
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
            state_dict = _convert_musk(state_dict, model)
            state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: random initialization for musk foundation model")
    else:
        print("Warning: random initialization for musk foundation model")

    return model

def chief(img_size, pretrained=False, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """ CTransPath foundation model
    """
    model = timm.create_model(
        model_name="swin_tiny_patch4_window7_224",
        img_size=img_size,
        num_classes=0,
        embed_layer=ConvStem,  # defined above
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
        pretrained=False,
    )

    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path)["model"]
        else:
            print("Warning: random initialization for chief-ctranspath foundation model")
        state_dict = adapt_checkpoint_ctranspath(state_dict)
        incompatible_keys = model.load_state_dict(state_dict, strict=True)
        print("加载完成，以下是详细信息：\n")

        # 打印缺失的键 (在 model_b 中存在，但在 pretrained_state_dict 中不存在)
        if incompatible_keys.missing_keys:
            print("缺失的键 (Missing keys):")
            for key in incompatible_keys.missing_keys:
                print(f"\t- {key}")

        # 打印意外的键 (在 pretrained_state_dict 中存在，但在 model_b 中不存在)
        if incompatible_keys.unexpected_keys:
            print("意外的键 (Unexpected keys):")
            for key in incompatible_keys.unexpected_keys:
                print(f"\t- {key}")
        else:
            print("没有意外的键。")
    else:
        print("Warning: random initialization for ctranspath foundation model")
    return model

def keep(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    model = timm.create_model('vit_large_patch16_224', img_size=img_size, init_values=1e-05, dynamic_img_size=False,
                              num_classes=0,
                              global_pool=global_pool, drop_path_rate=drop_path_rate, pretrained=False)

    if pretrained or ckpt_path:
        if pretrained:
            state_dict = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True).state_dict()
        else:
            print("Warning: random initialization for keep foundation model")

        state_dict = _convert_keep(state_dict, model, prefix='visual.')
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: random initialization for keep foundation model")

    return model

def conchv1_5(img_size, pretrained=False, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    # same to uni, except the input size is 448 (pos_embed)
    model = VisionTransformer(img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, pre_norm=False,
                      num_classes=0, dynamic_img_size=False, init_values=1.0, drop_path_rate=drop_path_rate, global_pool=global_pool)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location='cpu')
        else:

            print("Warning: random initialization for conchv1_5 foundation model")

        state_dict = _convert_conch(state_dict, model, prefix='trunk.') # coca arch.
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: random initialization for conchv1_5 foundation model")

    return model



def conch(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    # https://github.com/mahmoodlab/CONCH/blob/main/conch/open_clip_custom/coca_model.py
    # conch default input: 448; no pre_norm
    model = VisionTransformer(img_size=img_size,
                              patch_size=16, pre_norm=False, num_classes=0,
                              dynamic_img_size=False, global_pool=global_pool)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location='cpu')
        else:
            from conch.open_clip_custom import create_model_from_pretrained
            # default input size: 448
            clip_model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch",
                                                                  hf_auth_token=os.environ.get("HF_TOKEN"))

            state_dict = clip_model.state_dict()
        state_dict = _convert_conch(state_dict, model, prefix='visual.trunk.') # coca arch.
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict, strict=False)

    return model

def gpfm(img_size, pretrained=False, ckpt_path='', drop_path_rate=0., global_pool="") -> VisionTransformer:
    # based on dinov2
    model = timm.create_model('vit_large_patch14_224', img_size=img_size, pretrained=False, num_classes=0,
                              global_pool=global_pool, init_values=1e-5, drop_path_rate=drop_path_rate)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict_raw = torch.load(ckpt_path, map_location='cpu')['teacher']
            state_dict = {k.removeprefix('backbone.'): v for k, v in state_dict_raw.items()}
            state_dict = _convert_gpfm(state_dict, model)
            state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: random initialization for gpfm foundation model")

    return model

def pathgen(img_size, pretrained=False, ckpt_path='./pretrain-ckpts/pathgen-clip.pt', drop_path_rate=0., global_pool="") -> VisionTransformer:
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L1246
    # clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='path/pathgen-clip.pt')
    model = timm.create_model(
        model_name='vit_base_patch16_224', img_size=img_size,
        patch_size=16, pre_norm=True, num_classes=0, pretrained=False,
        dynamic_img_size=False, global_pool=global_pool, drop_path_rate=drop_path_rate) # pre_norm: usually clip-based model are true
    if pretrained or ckpt_path:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict = _convert_openai_clip(state_dict, model, prefix='visual.')
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict, strict=False)

    return model

def h0mini(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """
    H0-mini: https://huggingface.co/bioptimus/H0-mini/blob/main/config.json
    """
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2", img_size=img_size,
        drop_path_rate=drop_path_rate, num_classes=0, patch_size=14,
        global_pool=global_pool, pretrained=False, init_values=1e-5,
        dynamic_img_size=False, reg_tokens=4, mlp_ratio=5.3334, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,)

    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["h0-mini"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)

        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for h0-mini foundation model")

    return model


def virchow2(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool=""):
    # model = timm.create_model(
    #     "hf-hub:paige-ai/Virchow2", img_size=img_size, pretrained=False, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
    # )
    model = timm.create_model(
        "vit_huge_patch14_224", img_size=img_size, init_values=1e-5, pretrained=False, num_classes=0,
    reg_tokens=4, mlp_ratio=5.3375, global_pool=global_pool, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU
    )
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["virchow2"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for virchow2 foundation model")

    return model

def uni(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:

    #  https://huggingface.co/MahmoodLab/UNI/blob/main/config.json
    model = timm.create_model(
        model_name='vit_large_patch16_224', img_size=img_size,
        patch_size=16, depth=24, num_heads=16,
        drop_path_rate=drop_path_rate, init_values=1.0,
        embed_dim=1024, num_classes=0,
        global_pool=global_pool, pretrained=False,
        dynamic_img_size=False)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["uni"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for univ2 foundation model")

    return model

def univ2(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """ Univ2 foundation model
    """
    model = timm.create_model(
        model_name='vit_giant_patch14_224', img_size=img_size, 
        patch_size=14, depth=24, num_heads=24,
        drop_path_rate=drop_path_rate, init_values=1e-5, 
        embed_dim=1536, mlp_ratio=2.66667*2,
        num_classes=0,  no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, 
        act_layer=torch.nn.SiLU, reg_tokens=8,
        global_pool=global_pool, pretrained=False,
        dynamic_img_size=False)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["univ2"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for univ2 foundation model")
    return model


def hoptimus0(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """ Hoptimus foundation model
    """
    model = timm.create_model(
        "vit_giant_patch14_reg4_dinov2", img_size=img_size,
        drop_path_rate=drop_path_rate, num_classes=0,
        global_pool=global_pool, pretrained=False, init_values=1e-5,
        dynamic_img_size=False)

    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["hoptimus0"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for hoptimus0 foundation model")
    
    return model

def sp85m(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """ SP85m foundation model
    """
    model = timm.create_model(
        "vit_base_patch16_224", img_size=img_size,
        num_classes=0, drop_path_rate=drop_path_rate,
        global_pool=global_pool, pretrained=False,
        dynamic_img_size=False)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["sp85m"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = {k.replace("encoder.", ""): v for k,v in state_dict.items()}
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size=img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for sp85m foundation model")
    return model


def provgigapath(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """ ProvGigapath foundation model
    """
    model = timm.create_model(
        "vit_giant_patch14_dinov2", img_size=img_size,
        num_classes=0, patch_size=16, global_pool=global_pool,
        drop_path_rate=drop_path_rate, pretrained=False,
        init_values=1e-5, dynamic_img_size=False)
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["provgigapath"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size=img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for provgigapath foundation model")
    return model


def phikonv2(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """ Phikon-v2 foundation model
    """
    model = timm.create_model(
        "vit_large_patch14_dinov2", img_size=img_size,
        num_classes=0, patch_size=16, global_pool=global_pool,
        drop_path_rate=drop_path_rate, pretrained=False,
        dynamic_img_size=False)

    if pretrained or ckpt_path:
        depth = len(model.blocks)
        embed_dim = model.embed_dim
        if ckpt_path:
            state_dict = load_file(ckpt_path)
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["phikonv2"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        state_dict = hf2timm_checkpoint_conversion(state_dict, depth, embed_dim)
        state_dict = resize_pos_embed_statedict(state_dict, model, img_size=img_size)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for phikonv2 foundation model")
    return model


def restnet50_lunit_swav(img_size=None, pretrained=True, ckpt_path=None, drop_rate=0.) -> ResNet:
    """ Resnet50 Lunit Swav foundation model
    """
    model = timm.create_model(
        model_name="resnet50",
        num_classes=0,
        drop_rate=drop_rate,
        pretrained=True,
        )

    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = load_file(ckpt_path)
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["restnet50_lunit_swav"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("Warning: random initialization for restnet50_lunit_swav foundation model")
    return model


def ctranspath(img_size, pretrained=True, ckpt_path=None, drop_path_rate=0., global_pool="") -> VisionTransformer:
    """ CTransPath foundation model
    """
    model = timm.create_model(
        model_name="swin_tiny_patch4_window7_224",
        img_size=img_size,
        num_classes=0,
        embed_layer=ConvStem, #  defined above
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
        pretrained=False,
        )
    
    if pretrained or ckpt_path:
        if ckpt_path:
            state_dict = torch.load(ckpt_path)["model"]
        else:
            _, model_id = parse_model_name(FOUNDATION_HF_CKPT_REGISTRY["ctranspath"])
            state_dict = load_state_dict_from_hf(model_id, weights_only=True, filename="ctranspath.pth")["model"]
        state_dict = adapt_checkpoint_ctranspath(state_dict)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("Warning: random initialization for ctranspath foundation model")
    return model





def resize_pos_embed_statedict(state_dict, model, img_size):
    old_pos_embed = state_dict['pos_embed']
    pos_embed = resample_abs_pos_embed(
        old_pos_embed,
        new_size=model.patch_embed.grid_size,
        num_prefix_tokens=0 if model.no_embed_class \
            else model.num_prefix_tokens,
    )
    state_dict['pos_embed'] = pos_embed
    return state_dict


def hf2timm_checkpoint_conversion(state_dict, depth, embed_dim, use_swiglu_ffn=False):
    #https://github.com/huggingface/transformers/blob/76048be419b4ca90b58330a2564ab2fc0174a6b4/src/transformers/models/dinov2/convert_dinov2_to_hf.py#L146
    rename_keys = create_rename_keys(depth=depth, use_swiglu_ffn=use_swiglu_ffn)
    for dest, src in rename_keys:
        rename_key(state_dict, src, dest)
    state_dict = convert_hf_qkv_to_timm(state_dict, depth, embed_dim)

    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "weights_in" in key:
            key = key.replace("weights_in", "w12")
        if "weights_out" in key:
            key = key.replace("weights_out", "w3")
        state_dict[key] = val

    state_dict.pop("mask_token")
    return state_dict


def create_rename_keys(depth, use_swiglu_ffn):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("mask_token", "embeddings.mask_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))

    for i in range(depth):
        # layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"))
        # MLP
        if use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"))
        # layerscale
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"))
        # attention projection layer
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias"))

    # final layernorm
    rename_keys.append(("norm.weight", "layernorm.weight"))
    rename_keys.append(("norm.bias", "layernorm.bias"))

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def read_in_q_k_v(state_dict, depth, embed_dim):
    for i in range(depth):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: embed_dim, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: embed_dim]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            embed_dim : embed_dim * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            embed_dim : embed_dim * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-embed_dim :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-embed_dim :]


def convert_hf_qkv_to_timm(state_dict, depth, embed_dim):
    for i in range(depth):
        # Extract query, key, and value weights
        query_weight = state_dict.pop(f"encoder.layer.{i}.attention.attention.query.weight")
        key_weight = state_dict.pop(f"encoder.layer.{i}.attention.attention.key.weight")
        value_weight = state_dict.pop(f"encoder.layer.{i}.attention.attention.value.weight")

        # Concatenate weights into TIMM's format
        in_proj_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)

        # Extract query, key, and value biases
        query_bias = state_dict.pop(f"encoder.layer.{i}.attention.attention.query.bias")
        key_bias = state_dict.pop(f"encoder.layer.{i}.attention.attention.key.bias")
        value_bias = state_dict.pop(f"encoder.layer.{i}.attention.attention.value.bias")

        # Concatenate biases into TIMM's format
        in_proj_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)

        # Store in TIMM format
        state_dict[f"blocks.{i}.attn.qkv.weight"] = in_proj_weight
        state_dict[f"blocks.{i}.attn.qkv.bias"] = in_proj_bias

    return state_dict


class ConvStem(nn.Module):
  """Custom Patch Embed Layer.

  Adapted from https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py#L6-L44
  """

  def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, **kwargs):
    super().__init__()

    # Check input constraints
    assert patch_size == 4, "Patch size must be 4"
    assert embed_dim % 8 == 0, "Embedding dimension must be a multiple of 8"

    img_size = to_2tuple(img_size)
    patch_size = to_2tuple(patch_size)

    self.img_size = img_size
    self.patch_size = patch_size
    self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    self.num_patches = self.grid_size[0] * self.grid_size[1]

    # Create stem network
    stem = []
    input_dim, output_dim = 3, embed_dim // 8
    for l in range(2):
      stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
      stem.append(nn.BatchNorm2d(output_dim))
      stem.append(nn.ReLU(inplace=True))
      input_dim = output_dim
      output_dim *= 2
    stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
    self.proj = nn.Sequential(*stem)

    # Apply normalization layer (if provided)
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    B, C, H, W = x.shape

    # Check input image size
    assert H == self.img_size[0] and W == self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

    x = self.proj(x)
    x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
    x = self.norm(x)
    return x


def adapt_checkpoint_ctranspath(state_dict):

  new_state_dict = {}
  for k, v in state_dict.items():
      if ".downsample.norm" in k or "downsample.reduction" in k:
          k_split = k.split(".")
          k_split[1] = str(int(k_split[1]) + 1)
          new_k = ".".join(k_split)
      elif 'relative_position_index' in k or 'attn_mask' in k:
          continue
      else:
          new_k = k
      new_state_dict[new_k] = v
  return new_state_dict


def _convert_openai_clip(
        state_dict: dict[str, torch.Tensor],
        model: VisionTransformer,
        prefix: str = 'visual.',
) -> dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])
        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
        out_dict[k] = v
    return out_dict

def _convert_conch(
        state_dict: dict[str, torch.Tensor],
        model: VisionTransformer,
        prefix: str = 'visual.trunk.',
) -> dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        out_dict[k] = v

    return out_dict

def _convert_gpfm(
        state_dict: dict[str, torch.Tensor],
        model: VisionTransformer,
) -> dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    pattern = re.compile(r'^blocks\.(\d+)\.(.*)')
    for k, v in state_dict.items():
        if pattern.match(k):
            new_key = f"blocks.{pattern.match(k).group(2)}"
            out_dict[new_key] = v
        else:
            out_dict[k] = v
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
    return out_dict

def _convert_keep(
        state_dict: dict[str, torch.Tensor],
        model: VisionTransformer,
        prefix: str = 'visual.',
) -> dict[str, torch.Tensor]:
    out_dict = {}

    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        out_dict[k] = v
    return out_dict

def _convert_musk(state_dict: dict, model):
    """
    Turn a BEiT-3 checkpoint into a standard VisionTransformer state-dict.
    """
    import re
    state_dict = state_dict.get("model", state_dict)  # unwrap if needed

    # Prune unused
    for k in ("beit3.text_embed.weight", "beit3.vision_embed.mask_token"):
        state_dict.pop(k, None)

    # Key renaming rules
    rules = [
        (r"beit3\.", ""),
        (r"vision_embed\.cls_token", "cls_token"),
        (r"vision_embed\.",          "patch_embed."),
        (r"embed_positions\.",       "pos_embed."),
        (r"encoder\.", ""),
        (r"layers\.", "blocks."),
        (r"ffn_layernorm\.", "norm."), (r"ffn\.", "mlp."),
        (r"self_attn_layer_norm\.", "norm1."), (r"self_attn\.", "attn."),
        (r"final_layer_norm\.", "norm2."),
        (r"inner_attn_ln", "norm"),
        (r"out_proj", "proj"),
        (r"\.A\.", "."),
        (r"layer_norm.", "norm."),  # add 0817: final layer norm layer
    ]

    # First pass, rename keys
    tmp = {}
    for k, v in state_dict.items():
        if ".B." in k:
            continue  # use branch-A only
        for old, new in rules:
            k = re.sub(old, new, k)
        if k == "pos_embed.weight":
            # strip first two positions, [1, N+1, D]
            tmp["pos_embed"] = v[2:].unsqueeze(0)
        else:
            tmp[k] = v

    # Second pass, fuse q, k, v
    out, buf = {}, {}
    pat = re.compile(r"blocks\.(\d+)\.attn\.(q|k|v)_proj\.(weight|bias)$")
    for k, v in tmp.items():
        m = pat.fullmatch(k)
        if not m:  # anything not q/k/v -> copy through
            out[k] = v
            continue

        blk, which, kind = m.groups()  # block idx, 'q'/'k'/'v', 'weight'/'bias'
        stash = buf.setdefault((blk, kind), {})  # Gather by block & param type
        stash[which] = v
        if len(stash) == 3:  # Have q, k, v -> concatenate
            out[f"blocks.{blk}.attn.qkv.{kind}"] = torch.cat(
                [stash['q'], stash['k'], stash['v']], dim=0
            )

    return out

def _convert_omiclip(
        state_dict: dict[str, torch.Tensor],
        model: VisionTransformer,
        prefix: str = 'visual.',
) -> dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
        out_dict[k] = v
    return out_dict

FOUNDATION_MODEL_REGISTRY = {
    "univ2": univ2,
    "hoptimus0": hoptimus0,
    "sp85m": sp85m,
    "provgigapath": provgigapath,
    "phikonv2": phikonv2,
    "restnet50_lunit_swav": restnet50_lunit_swav,
    "ctranspath": ctranspath,
    'virchow2': virchow2,
    'h0-mini': h0mini,
    'pathgen': pathgen,
    'conch': conch,
    "uni": uni,
    "conchv1_5": conchv1_5,
    "gpfm": gpfm,
    "keep": keep,
    "chief": chief,
    "musk": musk,
    "omiclip": omiclip,
}
