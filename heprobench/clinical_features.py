from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as functional


def _decode_strings(values: Any) -> list[str]:
    raw = values.tolist() if hasattr(values, "tolist") else list(values)
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw]


def _safe_channel_filename(index: int, name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "channel"
    return f"{index:03d}_{normalized}.npy"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5_payload(path: Path) -> tuple[str, Any, list[str], list[str]]:
    handle = h5py.File(path, "r")
    try:
        data = handle["data"] if "data" in handle else handle
        if "pred" not in data:
            raise ValueError(f"Missing data/pred in {path}")
        predictions = data["pred"]
        if predictions.ndim != 4:
            raise ValueError(f"Expected predictions [N,H,W,C], got {predictions.shape} in {path}")
        if "channel_names" in data:
            channel_names = _decode_strings(data["channel_names"][()])
        else:
            channel_names = [f"channel_{index}" for index in range(predictions.shape[-1])]
        if len(channel_names) != predictions.shape[-1]:
            raise ValueError(f"Channel-name count does not match predictions in {path}")
        if "image_paths" in data:
            patch_ids = [Path(item).stem for item in _decode_strings(data["image_paths"][()])]
        elif "rows" in data and "cols" in data:
            patch_ids = [
                f"r{int(row)}_c{int(col)}"
                for row, col in zip(np.asarray(data["rows"]), np.asarray(data["cols"]))
            ]
        else:
            patch_ids = [f"patch_{index:06d}" for index in range(predictions.shape[0])]
        slide_name_value = handle.attrs.get("slide_name")
        if slide_name_value is None and "meta" in handle:
            slide_name_value = handle["meta"].attrs.get("slide_name")
        slide_name = str(slide_name_value or path.stem)
        slide_name = slide_name.replace("/", "_").replace("\\", "_")
        return slide_name, handle, channel_names, patch_ids
    except Exception:
        handle.close()
        raise


def _summary_features(images: np.ndarray) -> np.ndarray:
    flat = images.astype(np.float32).reshape(images.shape[0], -1) / 255.0
    return np.stack(
        [flat.mean(axis=1), flat.std(axis=1), flat.min(axis=1), flat.max(axis=1)],
        axis=1,
    ).astype(np.float32)


def _load_dinov2(extractor_cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    repository = str(extractor_cfg.get("repository", "facebookresearch/dinov2"))
    revision = str(extractor_cfg.get("revision", "main"))
    model_name = str(extractor_cfg.get("model", "dinov2_vits14"))
    model = torch.hub.load(f"{repository}:{revision}", model_name, trust_repo=True)
    expected_checksum = str(extractor_cfg.get("weights_sha256") or "").lower()
    checkpoint_name = str(extractor_cfg.get("checkpoint_filename", "dinov2_vits14_pretrain.pth"))
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if expected_checksum:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot verify configured DINOv2 checkpoint: {checkpoint_path}")
        actual = _file_sha256(checkpoint_path)
        if actual != expected_checksum:
            raise ValueError(
                f"DINOv2 checkpoint checksum mismatch: expected {expected_checksum}, got {actual}"
            )
    return model.eval().to(device)


def extract_virtual_features(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    extractor_cfg = feature_cfg.get("extractor", {})
    if not isinstance(extractor_cfg, dict):
        raise ValueError("features.extractor must be an object")
    input_dir = _path(data_cfg["prediction_h5_dir"], source_path)
    output_root = _path(feature_cfg["root"], source_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Clinical HDF5 prediction directory not found: {input_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    strategy = str(extractor_cfg.get("name", "dinov2_vits14")).lower()
    runtime = config.get("runtime", {})
    configured_device = runtime.get("device")
    if configured_device is None:
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(configured_device))
    model = _load_dinov2(extractor_cfg, device) if strategy == "dinov2_vits14" else None
    if strategy not in {"dinov2_vits14", "summary_stats"}:
        raise ValueError("features.extractor.name must be 'dinov2_vits14' or 'summary_stats'")
    input_size = int(extractor_cfg.get("input_size", 224))
    batch_size = int(extractor_cfg.get("batch_size", 128))
    overwrite = bool(extractor_cfg.get("overwrite", False))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    slides: list[dict[str, Any]] = []
    for h5_path in sorted(input_dir.glob("*.h5")):
        slide_name, handle, channel_names, patch_ids = _h5_payload(h5_path)
        try:
            predictions = (handle["data"] if "data" in handle else handle)["pred"]
            slide_dir = output_root / slide_name
            output_path = slide_dir / str(feature_cfg.get("virtual_channels_file", "virtual_channels.npy"))
            if output_path.exists() and not overwrite:
                slides.append({"slide_id": slide_name, "status": "existing", "path": str(output_path)})
                continue
            slide_dir.mkdir(parents=True, exist_ok=True)
            channel_features: list[np.ndarray] = []
            channel_index: list[dict[str, Any]] = []
            for channel_index_value, channel_name in enumerate(channel_names):
                pieces: list[np.ndarray] = []
                for start in range(0, predictions.shape[0], batch_size):
                    end = min(predictions.shape[0], start + batch_size)
                    images = np.asarray(predictions[start:end, :, :, channel_index_value], dtype=np.float32)
                    if strategy == "summary_stats":
                        pieces.append(_summary_features(images))
                        continue
                    tensor = torch.from_numpy(images).to(device).unsqueeze(1).repeat(1, 3, 1, 1)
                    if tensor.shape[-2:] != (input_size, input_size):
                        tensor = functional.interpolate(
                            tensor,
                            size=(input_size, input_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                    tensor = (tensor / 255.0 - mean) / std
                    with torch.no_grad():
                        pieces.append(model(tensor).detach().cpu().numpy().astype(np.float32))
                channel_array = np.concatenate(pieces, axis=0)
                channel_features.append(channel_array)
                channel_file = _safe_channel_filename(channel_index_value, channel_name)
                np.save(slide_dir / channel_file, channel_array)
                channel_index.append(
                    {"index": channel_index_value, "name": channel_name, "file": channel_file}
                )
            stacked = np.stack(channel_features, axis=1).astype(np.float32)
            np.save(output_path, stacked)
            np.save(
                slide_dir / str(feature_cfg.get("virtual_aggregated_file", "virtual_aggregated.npy")),
                stacked.mean(axis=1),
            )
            (slide_dir / str(feature_cfg.get("patch_ids_file", "patch_ids.json"))).write_text(
                json.dumps(patch_ids, indent=2) + "\n",
                encoding="utf-8",
            )
            (slide_dir / "channels.json").write_text(
                json.dumps(channel_index, indent=2) + "\n",
                encoding="utf-8",
            )
            slides.append(
                {
                    "slide_id": slide_name,
                    "status": "written",
                    "n_patches": int(stacked.shape[0]),
                    "n_channels": int(stacked.shape[1]),
                    "feature_dim": int(stacked.shape[2]),
                    "path": str(output_path),
                }
            )
        finally:
            handle.close()
    if not slides:
        raise ValueError(f"No .h5 prediction files found under {input_dir}")
    return {"stage": "extract-features", "extractor": strategy, "slides": slides}


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32, copy=False)
    if path.suffix.lower() in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected tensor feature file: {path}")
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    raise ValueError(f"Unsupported feature file type: {path}")


def _read_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"Patch ID file must contain a JSON string list: {path}")
    if len(set(payload)) != len(payload):
        raise ValueError(f"Patch IDs must be unique: {path}")
    return payload


def align_he_features(config: dict[str, Any], source_path: Path) -> dict[str, Any]:
    feature_cfg = config["features"]
    output_root = _path(feature_cfg["root"], source_path)
    he_root = _path(feature_cfg["he_root"], source_path)
    he_pattern = str(feature_cfg.get("he_file_pattern", "{slide_id}_feats.pt"))
    he_ids_pattern = str(feature_cfg.get("he_patch_ids_pattern", "{slide_id}_patch_id.json"))
    patch_ids_file = str(feature_cfg.get("patch_ids_file", "patch_ids.json"))
    virtual_file = str(feature_cfg.get("virtual_channels_file", "virtual_channels.npy"))
    he_output_file = str(feature_cfg.get("he_aligned_file", "he_aligned.npy"))
    strict = bool(feature_cfg.get("strict_alignment", True))
    slides: list[dict[str, Any]] = []
    for slide_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        slide_id = slide_dir.name
        virtual_path = slide_dir / virtual_file
        virtual_ids_path = slide_dir / patch_ids_file
        if not virtual_path.is_file() or not virtual_ids_path.is_file():
            continue
        he_path = he_root / he_pattern.format(slide_id=slide_id)
        he_ids_path = he_root / he_ids_pattern.format(slide_id=slide_id)
        if not he_path.is_file() or not he_ids_path.is_file():
            raise FileNotFoundError(f"Missing H&E feature/patch IDs for slide {slide_id}")
        he = _load_array(he_path)
        virtual = np.load(virtual_path).astype(np.float32, copy=False)
        he_ids = _read_ids(he_ids_path)
        virtual_ids = _read_ids(virtual_ids_path)
        if len(he_ids) != len(he) or len(virtual_ids) != len(virtual):
            raise ValueError(f"Feature/patch-ID length mismatch for slide {slide_id}")
        he_lookup = {patch_id: index for index, patch_id in enumerate(he_ids)}
        kept_virtual: list[int] = []
        kept_he: list[int] = []
        kept_ids: list[str] = []
        for virtual_index, patch_id in enumerate(virtual_ids):
            if patch_id in he_lookup:
                kept_virtual.append(virtual_index)
                kept_he.append(he_lookup[patch_id])
                kept_ids.append(patch_id)
        missing = len(virtual_ids) - len(kept_ids)
        if strict and missing:
            raise ValueError(f"{slide_id}: {missing} virtual patches have no matching H&E feature")
        if not kept_ids:
            raise ValueError(f"{slide_id}: no aligned H&E/virtual patches")
        aligned_he = he[kept_he]
        aligned_virtual = virtual[kept_virtual]
        np.save(slide_dir / he_output_file, aligned_he)
        np.save(virtual_path, aligned_virtual)
        np.save(
            slide_dir / str(feature_cfg.get("virtual_aggregated_file", "virtual_aggregated.npy")),
            aligned_virtual.mean(axis=1),
        )
        (slide_dir / patch_ids_file).write_text(json.dumps(kept_ids, indent=2) + "\n", encoding="utf-8")
        slides.append(
            {
                "slide_id": slide_id,
                "n_aligned": len(kept_ids),
                "n_missing_he": missing,
                "he_dim": int(aligned_he.shape[-1]),
                "virtual_dim": int(aligned_virtual.shape[-1]),
            }
        )
    if not slides:
        raise ValueError(f"No extracted virtual slide directories found under {output_root}")
    return {"stage": "align-features", "slides": slides, "strict": strict}


def _path(value: str | Path, source_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source_path.parent / path
    return path.resolve()
