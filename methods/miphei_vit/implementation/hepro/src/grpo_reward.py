"""Reward functions for GRPO-style HE->mIF post-training."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _rank_1d(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def _pearson_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if (not torch.isfinite(denom)) or denom <= eps:
        return torch.zeros((), device=x.device, dtype=torch.float32)
    return torch.clamp((x * y).mean() / (denom + eps), min=-1.0, max=1.0)


def _spearman_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if x.numel() < 2 or y.numel() < 2:
        return torch.zeros((), device=x.device, dtype=torch.float32)
    rx = _rank_1d(x)
    ry = _rank_1d(y)
    return _pearson_1d(rx, ry, eps=eps)


def _corrcoef_features(features: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Compute marker-marker correlation matrix from [N_cells, N_markers]."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")

    n_cells = features.shape[0]
    if n_cells < 2:
        c = features.shape[1]
        return torch.eye(c, dtype=torch.float32, device=features.device)

    x = features.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x
    cov = cov / max(n_cells - 1, 1)

    std = torch.sqrt(torch.clamp(torch.diag(cov), min=0.0))
    denom = std[:, None] * std[None, :]
    corr = cov / (denom + eps)
    corr = torch.clamp(corr, min=-1.0, max=1.0)
    corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
    return corr


def _cell_means_from_mask(
    value_map: torch.Tensor,
    nuclei_mask: torch.Tensor,
    min_cell_area: int,
) -> torch.Tensor:
    """Aggregate per-cell mean marker expression.

    Args:
        value_map: [C, H, W]
        nuclei_mask: [H, W], integer mask (0 is background)
        min_cell_area: minimum number of pixels for a valid cell
    Returns:
        Tensor [N_cells, C]. If no valid cell, returns [0, C].
    """
    if value_map.ndim != 3:
        raise ValueError(f"value_map must be [C,H,W], got {value_map.shape}")
    if nuclei_mask.ndim != 2:
        raise ValueError(f"nuclei_mask must be [H,W], got {nuclei_mask.shape}")

    c = value_map.shape[0]
    flat_mask = nuclei_mask.reshape(-1).long()
    valid = flat_mask > 0

    if valid.sum() == 0:
        return value_map.new_zeros((0, c), dtype=torch.float32)

    labels = flat_mask[valid]
    unique_labels, inverse = torch.unique(labels, return_inverse=True)

    flat_vals = value_map.permute(1, 2, 0).reshape(-1, c)[valid]
    sums = value_map.new_zeros((unique_labels.numel(), c), dtype=torch.float32)
    sums.scatter_add_(0, inverse[:, None].expand(-1, c), flat_vals.float())

    counts = value_map.new_zeros((unique_labels.numel(),), dtype=torch.float32)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))

    keep = counts >= float(min_cell_area)
    if keep.sum() == 0:
        return value_map.new_zeros((0, c), dtype=torch.float32)

    means = sums[keep] / counts[keep, None]
    return means


def compute_patch_reward(
    pred_map: torch.Tensor,
    target_map: torch.Tensor,
    nuclei_mask: torch.Tensor,
    min_cell_area: int = 20,
    min_cells_per_patch: int = 10,
    w_coloc: float = 0.5,
    w_rank: float = 0.5,
    inputs_normalized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total/co-localization/rank rewards for a single patch."""
    pred = pred_map
    target = target_map

    if inputs_normalized:
        pred = (pred.clamp(-0.9, 0.9) + 0.9) / 1.8
        target = (target.clamp(-0.9, 0.9) + 0.9) / 1.8

    cell_pred = _cell_means_from_mask(pred, nuclei_mask, min_cell_area=min_cell_area)
    cell_target = _cell_means_from_mask(target, nuclei_mask, min_cell_area=min_cell_area)

    n_cells = min(cell_pred.shape[0], cell_target.shape[0])
    if n_cells < int(min_cells_per_patch):
        zero = pred.new_zeros((), dtype=torch.float32)
        return zero, zero, zero

    cell_pred = cell_pred[:n_cells]
    cell_target = cell_target[:n_cells]

    # Co-localization reward.
    corr_pred = _corrcoef_features(cell_pred)
    corr_target = _corrcoef_features(cell_target)
    n_markers = corr_pred.shape[0]
    if n_markers < 2:
        r_coloc = pred.new_zeros((), dtype=torch.float32)
    else:
        tri = torch.triu_indices(n_markers, n_markers, offset=1, device=pred.device)
        vec_pred = corr_pred[tri[0], tri[1]]
        vec_target = corr_target[tri[0], tri[1]]
        if vec_pred.numel() == 0:
            r_coloc = pred.new_zeros((), dtype=torch.float32)
        else:
            r_coloc = F.cosine_similarity(
                vec_pred.unsqueeze(0), vec_target.unsqueeze(0), dim=1
            ).squeeze(0)
            if not torch.isfinite(r_coloc):
                r_coloc = pred.new_zeros((), dtype=torch.float32)

    # Rank reward.
    rhos = []
    for idx in range(cell_pred.shape[1]):
        rho = _spearman_1d(cell_pred[:, idx], cell_target[:, idx])
        rhos.append(rho)
    if len(rhos) == 0:
        r_rank = pred.new_zeros((), dtype=torch.float32)
    else:
        r_rank = torch.stack(rhos).mean()
        r_rank = torch.where(torch.isfinite(r_rank), r_rank, torch.zeros_like(r_rank))

    reward = (float(w_coloc) * r_coloc) + (float(w_rank) * r_rank)
    return reward, r_coloc, r_rank


def compute_group_rewards(
    pred_group: torch.Tensor,
    target_group: torch.Tensor,
    nuclei_masks: torch.Tensor,
    min_cell_area: int = 20,
    min_cells_per_patch: int = 10,
    w_coloc: float = 0.5,
    w_rank: float = 0.5,
    inputs_normalized: bool = True,
    return_components: bool = False,
):
    """Compute rewards for [B,G,C,H,W] tensors."""
    if pred_group.ndim != 5:
        raise ValueError(f"pred_group must be [B,G,C,H,W], got {pred_group.shape}")
    if target_group.shape != pred_group.shape:
        raise ValueError(
            f"target_group shape mismatch: {target_group.shape} vs {pred_group.shape}"
        )
    if nuclei_masks.ndim != 3:
        raise ValueError(f"nuclei_masks must be [B,H,W], got {nuclei_masks.shape}")

    bsz, group = pred_group.shape[:2]
    rewards = pred_group.new_zeros((bsz, group), dtype=torch.float32)
    coloc = pred_group.new_zeros((bsz, group), dtype=torch.float32)
    rank = pred_group.new_zeros((bsz, group), dtype=torch.float32)

    for b in range(bsz):
        mask_b = nuclei_masks[b]
        for g in range(group):
            r, rc, rr = compute_patch_reward(
                pred_map=pred_group[b, g],
                target_map=target_group[b, g],
                nuclei_mask=mask_b,
                min_cell_area=min_cell_area,
                min_cells_per_patch=min_cells_per_patch,
                w_coloc=w_coloc,
                w_rank=w_rank,
                inputs_normalized=inputs_normalized,
            )
            rewards[b, g] = r
            coloc[b, g] = rc
            rank[b, g] = rr

    if return_components:
        return rewards, coloc, rank
    return rewards
