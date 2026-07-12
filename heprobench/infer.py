from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import load_dataset_config, load_method_config
from .io_h5 import write_slide_h5
from .schema import PatchRecord, group_by_slide, load_channel_names, load_records
from methods import build_method


class PatchDataset(Dataset):
    def __init__(self, records: list[PatchRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        with Image.open(rec.image_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return {"image": tensor, "index": index}


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str | None, device: torch.device) -> None:
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
        )


def _predict_records(
    records: list[PatchRecord],
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = PatchDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[tuple[int, np.ndarray]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pred = model(images).clamp(0.0, 1.0)
            pred_u8 = (pred * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for idx, arr in zip(batch["index"].tolist(), pred_u8):
                outputs.append((int(idx), arr))
    outputs.sort(key=lambda x: x[0])
    return np.stack([arr for _, arr in outputs], axis=0)


def run_inference(
    config_path: str | Path,
    method: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
) -> dict:
    dataset_cfg, dataset_config_path = load_dataset_config(config_path)
    method_cfg, _ = load_method_config(method, dataset_config_path)
    dataset = dataset_cfg["dataset"]
    method_block = method_cfg["method"]
    channel_names = load_channel_names(dataset["channel_names"])
    records = load_records(dataset)
    grouped = group_by_slide(records)

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_method(method_block, out_channels=len(channel_names)).to(torch_device)
    _load_checkpoint(model, method_block.get("checkpoint"), torch_device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(method_block.get("batch_size", 4))
    written = []
    for slide_name, slide_records in grouped.items():
        pred = _predict_records(slide_records, model, batch_size=batch_size, device=torch_device)
        h5_path = output_dir / f"{slide_name}.h5"
        write_slide_h5(
            h5_path,
            predictions=pred,
            records=slide_records,
            channel_names=channel_names,
            attrs={
                "method_name": method_block.get("name", "unknown"),
                "method_display_name": method_block.get("display_name", ""),
                "encoder_name": method_block.get("encoder_name", ""),
                "split": dataset.get("split", "demo"),
                "patch_size": int(dataset.get("patch_size", 256)),
            },
        )
        written.append(str(h5_path))
    return {
        "method": method_block.get("name", "unknown"),
        "encoder_name": method_block.get("encoder_name", ""),
        "slides": len(written),
        "output_dir": str(output_dir),
        "files": written,
    }

