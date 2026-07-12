from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import load_dataset_config, load_method_config
from .schema import load_channel_names, load_records
from methods import build_method


class TrainPatchDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        with Image.open(rec.image_path) as img:
            image = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        if rec.target_path is None:
            raise ValueError(f"Missing target for training record: {rec}")
        target = np.load(rec.target_path).astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "target": torch.from_numpy(target).permute(2, 0, 1),
        }


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str | None, device: torch.device) -> None:
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        return
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=False)


def run_training(
    config_path: str | Path,
    method: str | Path,
    output_checkpoint: str | Path,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    dataset_cfg, dataset_config_path = load_dataset_config(config_path)
    method_cfg, _ = load_method_config(method, dataset_config_path)
    dataset = dataset_cfg["dataset"]
    method_block = method_cfg["method"]
    channel_names = load_channel_names(dataset["channel_names"])
    records = load_records(dataset)

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_method(method_block, out_channels=len(channel_names)).to(torch_device)
    _load_checkpoint(model, method_block.get("checkpoint"), torch_device)

    loader = DataLoader(
        TrainPatchDataset(records),
        batch_size=int(method_block.get("batch_size", 4)),
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0
    for _epoch in range(int(epochs)):
        for batch in loader:
            image = batch["image"].to(torch_device)
            target = batch["target"].to(torch_device)
            pred = model(image)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "method": method_block,
            "epochs": int(epochs),
            "loss": last_loss,
            "note": "Tiny checkpoint trained on synthetic review demo data.",
        },
        output_checkpoint,
    )
    return {
        "method": method_block.get("name", "unknown"),
        "encoder_name": method_block.get("encoder_name", ""),
        "epochs": int(epochs),
        "loss": last_loss,
        "checkpoint": str(output_checkpoint),
    }

