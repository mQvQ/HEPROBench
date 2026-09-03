from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from os.path import join
from typing import Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from hex.hex_architecture_fds import CustomModelFDS
from hex.utils import CustomDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, required=True)
    p.add_argument("--phase", type=str, default="test", choices=["valid", "test", "infer_valid_test"])
    p.add_argument("--panel_key", type=str, default="default")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--musk_img_size", type=int, default=256)
    p.add_argument("--norm", choices=["paper", "imagenet"], default="imagenet")
    p.add_argument("--save_predictions", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.norm == "paper":
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

        normalize = transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = CustomDataset(dataroot=args.dataroot, phase=args.phase, panel_key=args.panel_key, transform=transform)
    num_outputs = int(np.asarray(dataset[0][1]).shape[0])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = CustomModelFDS(visual_output_dim=1024, num_outputs=num_outputs, musk_img_size=args.musk_img_size).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    model.training_status = False

    preds_all = []
    labels_all = []
    he_paths_all = []
    if_paths_all = []

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), autocast_ctx:
        for inputs, labels, he_path, if_path in loader:
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float16 if device.type == "cuda" else torch.float32)
            labels_t = labels.to(device, non_blocking=True, dtype=torch.float32)
            outputs, _ = model(inputs, labels_t, epoch=0)
            preds_all.append(outputs.to(dtype=torch.float32).cpu().numpy())
            labels_all.append(labels_t.cpu().numpy())
            he_paths_all.extend(list(he_path))
            if_paths_all.extend(list(if_path))

    preds = np.concatenate(preds_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)

    mse_per_marker = np.nanmean((labels_np - preds) ** 2, axis=0)
    rmse_per_marker = np.sqrt(mse_per_marker)
    metrics = {
        "mse_avg": float(np.nanmean(mse_per_marker)),
        "rmse_avg": float(np.nanmean(rmse_per_marker)),
    }

    metrics_path = join(args.save_dir, f"metrics_{args.phase}.json")
    try:
        import json

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

    if args.save_predictions:
        df = pd.DataFrame(
            {
                "he_path": he_paths_all,
                "if_path": if_paths_all,
            }
        )
        for j in range(num_outputs):
            df[f"pred_{j}"] = preds[:, j]
            df[f"label_{j}"] = labels_np[:, j]
        df.to_csv(join(args.save_dir, f"predictions_{args.phase}.csv"), index=False)


if __name__ == "__main__":
    main()
