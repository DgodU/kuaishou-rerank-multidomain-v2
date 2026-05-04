from __future__ import annotations

import torch
import torch.nn as nn


class CalibrationHead(nn.Module):
    def __init__(self, dense_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        self.dense_dim = int(dense_dim)
        if self.dense_dim <= 0:
            self.net = None
            return
        layers: list[nn.Module] = [nn.LayerNorm(self.dense_dim)]
        in_dim = self.dense_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.GELU(), nn.Dropout(dropout)])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, dense_features: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.net is None or dense_features is None:
            return torch.zeros(batch_size, device=device, dtype=dtype)
        return self.net(dense_features.to(device=device, dtype=dtype)).squeeze(-1)
