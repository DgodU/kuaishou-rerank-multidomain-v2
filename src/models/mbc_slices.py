from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


# MBC 语义切片头：融合 interest/target/domain/user/behavior_side 等切片，输出辅助残差 logit。
class MBCSemanticHead(nn.Module):
    def __init__(self, slice_dims: Dict[str, int], branch_dim: int = 128, fusion_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.slice_names = list(slice_dims.keys())
        input_dim = sum(slice_dims.values())
        # 简单 Deep 分支学习不同语义切片之间的组合关系。
        self.deep = nn.Sequential(
            nn.Linear(input_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, fusion_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(fusion_dim, 1)

    def forward(self, feature_slices: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # 按固定切片顺序拼接，保证训练和推理的输入语义一致。
        x = torch.cat([feature_slices[name] for name in self.slice_names], dim=-1)
        z = self.deep(x)
        logit = self.out(z).squeeze(-1)
        return z, logit
