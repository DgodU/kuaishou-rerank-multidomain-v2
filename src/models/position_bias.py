from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PositionBiasTower(nn.Module):
    def __init__(self, feature_maps: Dict, embedding_dim: int = 8, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        vocab_sizes = feature_maps.get("vocab_sizes", {})
        bucket_sizes = feature_maps.get("bucket_sizes", {})
        self.feature_names = [
            "position_bucket",
            "tab",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "user_active_degree",
            "target_category_l1",
            "target_category_l2",
            "target_video_type",
            "target_duration_bucket",
        ]
        sizes = {
            "position_bucket": int(bucket_sizes.get("position_bucket", 0)) + 1,
            "tab": int(vocab_sizes.get("tab", 6)),
            "hour_of_day": int(vocab_sizes.get("hour_of_day", 25)),
            "day_of_week": int(vocab_sizes.get("day_of_week", 8)),
            "is_weekend": int(vocab_sizes.get("is_weekend", 3)),
            "user_active_degree": int(vocab_sizes.get("user_active_degree", 2)),
            "target_category_l1": int(vocab_sizes.get("category_l1_id", 2)),
            "target_category_l2": int(vocab_sizes.get("category_l2_id", 2)),
            "target_video_type": int(vocab_sizes.get("video_type", 2)),
            "target_duration_bucket": int(vocab_sizes.get("duration_bucket", bucket_sizes.get("duration_bucket", 1) + 1)),
        }
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(max(2, size), embedding_dim, padding_idx=0) for name, size in sizes.items()}
        )
        layers: list[nn.Module] = []
        in_dim = embedding_dim * len(self.feature_names)
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = []
        for name in self.feature_names:
            value = batch.get(name)
            if value is None:
                first = next(iter(batch.values()))
                value = torch.zeros(first.size(0), device=first.device, dtype=torch.long)
            value = value.long().clamp_min(0)
            max_idx = self.embeddings[name].num_embeddings - 1
            value = value.clamp_max(max_idx)
            embs.append(self.embeddings[name](value))
        return self.mlp(torch.cat(embs, dim=-1)).squeeze(-1)
