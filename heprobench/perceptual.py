from __future__ import annotations

from typing import Any

import numpy as np


def _to_three_channel_tensor(values: np.ndarray) -> Any:
    """Convert uint8 [B,H,W,C] markers into [B*C,3,H,W] in [-1,1]."""

    import torch

    tensor = torch.from_numpy(values).to(torch.float32).permute(0, 3, 1, 2).contiguous()
    batch, channels, height, width = tensor.shape
    tensor = tensor.reshape(batch * channels, 1, height, width).repeat(1, 3, 1, 1)
    return tensor / 127.5 - 1.0


def _load_lpips(net: str, device: str) -> Any:
    try:
        import lpips
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError(
            "LPIPS evaluation requires lpips; install requirements-full.txt"
        ) from exc
    model = lpips.LPIPS(net=net)
    model.eval()
    return model.to(torch.device(device))


def _load_dists(device: str) -> Any:
    import sys
    from pathlib import Path

    import torch

    try:
        import DISTS_pytorch as dists_package
        from DISTS_pytorch import DISTS
    except ModuleNotFoundError:
        try:
            import dists_pytorch as dists_package
            from dists_pytorch import DISTS
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError(
                "DISTS evaluation requires DISTS-pytorch; install requirements-full.txt"
            ) from exc
    # DISTS-pytorch 0.1 hard-codes ``sys.prefix/weights.pt``. Some modern
    # installers keep the data file beside the package instead. Temporarily
    # point that legacy lookup at the package without copying or mutating it.
    package_weights = Path(dists_package.__file__).resolve().parent / "weights.pt"
    original_prefix = sys.prefix
    if package_weights.is_file() and not (Path(sys.prefix) / "weights.pt").is_file():
        sys.prefix = str(package_weights.parent)
    try:
        model = DISTS()
    finally:
        sys.prefix = original_prefix
    model.eval()
    return model.to(torch.device(device))


def perceptual_per_patch_channel(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    metrics: set[str],
    device: str,
    patch_batch: int,
    channel_chunk: int,
    lpips_net: str,
) -> dict[str, np.ndarray]:
    """Compute the original per-marker LPIPS/DISTS protocol on selected patches."""

    if predictions.shape != targets.shape or predictions.ndim != 4:
        raise ValueError("Perceptual inputs must have the same [N,H,W,C] shape")
    if patch_batch <= 0 or channel_chunk <= 0:
        raise ValueError("patch_batch and channel_chunk must be positive")

    import torch

    count, _height, _width, channels = predictions.shape
    requested = metrics.intersection({"lpips", "dists"})
    models = {
        "lpips": _load_lpips(lpips_net, device) if "lpips" in requested else None,
        "dists": _load_dists(device) if "dists" in requested else None,
    }
    output = {
        name: np.full((count, channels), np.nan, dtype=np.float32)
        for name in requested
    }

    for patch_start in range(0, count, patch_batch):
        patch_end = min(count, patch_start + patch_batch)
        batch_size = patch_end - patch_start
        for channel_start in range(0, channels, channel_chunk):
            channel_end = min(channels, channel_start + channel_chunk)
            chunk_size = channel_end - channel_start
            pred = _to_three_channel_tensor(
                predictions[patch_start:patch_end, ..., channel_start:channel_end]
            ).to(device)
            target = _to_three_channel_tensor(
                targets[patch_start:patch_end, ..., channel_start:channel_end]
            ).to(device)
            with torch.inference_mode():
                for name, model in models.items():
                    if model is None:
                        continue
                    value = model(pred, target)
                    if isinstance(value, (tuple, list)):
                        value = value[0]
                    output[name][patch_start:patch_end, channel_start:channel_end] = (
                        value.reshape(batch_size, chunk_size).detach().cpu().numpy().astype(np.float32)
                    )
    return output
