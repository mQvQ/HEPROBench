from __future__ import annotations

from typing import Any

import torch
from torch import nn


class GatedAttention(nn.Module):
    """Gated attention pooling used by the internal AMIL/MICA pipeline."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_c(self.attention_a(features) * self.attention_b(features))
        weights = torch.softmax(scores.transpose(0, 1), dim=1)
        pooled = torch.mm(weights, features)
        return pooled, scores.transpose(0, 1)


class AMIL(nn.Module):
    """Attention MIL head for a single H&E or virtual-protein feature bag."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        attention_dim: int = 256,
        projection_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.attention = GatedAttention(input_dim, attention_dim, dropout)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(projection_dim, output_dim)

    def forward(self, feature_bag: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled, attention = self.attention(feature_bag)
        logits = self.classifier(self.projection(pooled))
        return _prediction_payload(logits, attention=attention)


class MCAT(nn.Module):
    """Co-attention model used for aligned H&E + virtual-protein feature bags.

    The remote implementation used a copied PyTorch multi-head attention
    implementation. ``nn.MultiheadAttention`` has the same learned projections
    and is used here so the public implementation remains maintainable.
    """

    def __init__(
        self,
        *,
        he_input_dim: int,
        virtual_input_dim: int,
        output_dim: int,
        latent_dim: int = 256,
        attention_dim: int | None = None,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_feedforward_dim: int = 512,
        fusion: str = "concat",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if latent_dim % transformer_heads != 0:
            raise ValueError("model.latent_dim must be divisible by model.transformer_heads")
        if fusion not in {"concat", "bilinear"}:
            raise ValueError("model.fusion must be 'concat' or 'bilinear'")
        attention_dim = int(attention_dim or latent_dim)
        self.fusion = fusion
        self.he_projection = nn.Sequential(
            nn.Linear(he_input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.virtual_projection = nn.Sequential(
            nn.Linear(virtual_input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        channel_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.channel_transformer = nn.TransformerEncoder(channel_layer, num_layers=transformer_layers)
        self.coattention = nn.MultiheadAttention(latent_dim, num_heads=1, dropout=0.0, batch_first=False)
        bag_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=False,
        )
        self.bag_transformer = nn.TransformerEncoder(
            bag_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.bag_attention = GatedAttention(latent_dim, attention_dim, dropout)
        self.bag_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if fusion == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
            )
        else:
            reduced = max(1, latent_dim // 8)
            self.bilinear_he = nn.Sequential(nn.Linear(latent_dim, reduced), nn.ReLU())
            self.bilinear_virtual = nn.Sequential(nn.Linear(latent_dim, reduced), nn.ReLU())
            self.fusion_layer = nn.Sequential(
                nn.Linear(reduced * reduced, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
            )
        self.classifier = nn.Linear(latent_dim, output_dim)

    def _pool(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = self.bag_transformer(sequence.unsqueeze(1)).squeeze(1)
        pooled, attention = self.bag_attention(transformed)
        return self.bag_projection(pooled).squeeze(0), attention

    def forward(self, he_bag: torch.Tensor, virtual_bag: torch.Tensor) -> dict[str, torch.Tensor]:
        if he_bag.ndim != 2:
            raise ValueError(f"Expected H&E features [patch,dim], got {tuple(he_bag.shape)}")
        if virtual_bag.ndim not in {2, 3}:
            raise ValueError(
                f"Expected virtual features [patch,dim] or [patch,channel,dim], got {tuple(virtual_bag.shape)}"
            )
        if he_bag.shape[0] != virtual_bag.shape[0]:
            raise ValueError("H&E and virtual bags must be aligned and have the same patch count")

        he = self.he_projection(he_bag)
        if virtual_bag.ndim == 3:
            n_patches, n_channels, feature_dim = virtual_bag.shape
            virtual = self.virtual_projection(virtual_bag.reshape(n_patches * n_channels, feature_dim))
            virtual = self.channel_transformer(virtual.reshape(n_patches, n_channels, -1)).mean(dim=1)
        else:
            virtual = self.virtual_projection(virtual_bag)

        # Sequence first, one slide per batch, matching the internal MICA run.
        coattended_he, coattention = self.coattention(
            virtual.unsqueeze(1),
            he.unsqueeze(1),
            he.unsqueeze(1),
            need_weights=True,
            average_attn_weights=False,
        )
        he_pooled, he_attention = self._pool(coattended_he.squeeze(1))
        virtual_pooled, virtual_attention = self._pool(virtual)

        if self.fusion == "concat":
            fused = self.fusion_layer(torch.cat([he_pooled, virtual_pooled], dim=0))
        else:
            he_reduced = self.bilinear_he(he_pooled)
            virtual_reduced = self.bilinear_virtual(virtual_pooled)
            outer = torch.outer(he_reduced, virtual_reduced).reshape(1, -1)
            fused = self.fusion_layer(outer).squeeze(0)
        logits = self.classifier(fused).unsqueeze(0)
        return _prediction_payload(
            logits,
            attention={
                "coattention": coattention,
                "he": he_attention,
                "virtual": virtual_attention,
            },
        )


def _prediction_payload(
    logits: torch.Tensor,
    *,
    attention: torch.Tensor | dict[str, torch.Tensor],
) -> dict[str, Any]:
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return {
        "logits": logits,
        "hazards": hazards,
        "survival": survival,
        "attention": attention,
    }


def build_clinical_model(
    model_cfg: dict[str, Any],
    *,
    task_type: str,
    he_input_dim: int | None,
    virtual_input_dim: int | None,
    output_dim: int,
) -> nn.Module:
    model_name = str(model_cfg.get("name", "amil")).lower()
    dropout = float(model_cfg.get("dropout", 0.25))
    if model_name == "amil":
        input_modality = str(model_cfg.get("input_modality", "virtual")).lower()
        input_dim = he_input_dim if input_modality == "he" else virtual_input_dim
        if input_dim is None:
            raise ValueError(f"AMIL input dimension is unavailable for modality={input_modality!r}")
        return AMIL(
            input_dim=int(input_dim),
            output_dim=output_dim,
            attention_dim=int(model_cfg.get("attention_dim", 256)),
            projection_dim=int(model_cfg.get("projection_dim", 256)),
            dropout=dropout,
        )
    if model_name == "mcat":
        if he_input_dim is None or virtual_input_dim is None:
            raise ValueError("MCAT requires both H&E and virtual-protein features")
        return MCAT(
            he_input_dim=int(he_input_dim),
            virtual_input_dim=int(virtual_input_dim),
            output_dim=output_dim,
            latent_dim=int(model_cfg.get("latent_dim", 256)),
            attention_dim=int(model_cfg.get("attention_dim", model_cfg.get("latent_dim", 256))),
            transformer_heads=int(model_cfg.get("transformer_heads", 8)),
            transformer_layers=int(model_cfg.get("transformer_layers", 2)),
            transformer_feedforward_dim=int(model_cfg.get("transformer_feedforward_dim", 512)),
            fusion=str(model_cfg.get("fusion", "concat")),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported clinical model {model_name!r}; expected 'amil' or 'mcat'")


def forward_clinical_model(
    model: nn.Module,
    model_cfg: dict[str, Any],
    *,
    he: torch.Tensor | None,
    virtual: torch.Tensor | None,
) -> dict[str, Any]:
    model_name = str(model_cfg.get("name", "amil")).lower()
    if model_name == "amil":
        modality = str(model_cfg.get("input_modality", "virtual")).lower()
        features = he if modality == "he" else virtual
        if features is None:
            raise ValueError(f"Missing {modality} features for AMIL")
        return model(features)
    if he is None or virtual is None:
        raise ValueError("MCAT requires both H&E and virtual-protein features")
    return model(he, virtual)


def discrete_time_nll(
    hazards: torch.Tensor,
    survival: torch.Tensor,
    labels: torch.Tensor,
    censorship: torch.Tensor,
    *,
    alpha: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Negative log-likelihood from the MICA/MCAT discrete survival run.

    ``censorship`` uses the historical convention: 1 means censored and 0
    means the event was observed.
    """

    batch_size = len(labels)
    labels = labels.view(batch_size, 1).long()
    censorship = censorship.view(batch_size, 1).float()
    padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    uncensored_loss = -(1.0 - censorship) * (
        torch.log(torch.gather(padded, 1, labels).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, labels).clamp(min=eps))
    )
    censored_loss = -censorship * torch.log(
        torch.gather(padded, 1, labels + 1).clamp(min=eps)
    )
    negative_likelihood = censored_loss + uncensored_loss
    return ((1.0 - alpha) * negative_likelihood + alpha * uncensored_loss).mean()
