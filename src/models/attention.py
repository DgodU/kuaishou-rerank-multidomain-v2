from __future__ import annotations

import torch
import torch.nn as nn


# DIN 风格目标注意力：用 MLP 学习 target 与每个历史 item 的非线性匹配分数。
class DINStyleTargetAttention(nn.Module):
    def __init__(self, d_attn: int, attn_hidden_dim: int = 128, dropout: float = 0.1, use_side_bias: bool = True):
        super().__init__()
        self.use_side_bias = use_side_bias
        # 输入拼接 Q、K、Q-K、Q*K，是 DIN 中常见的目标-历史匹配特征。
        self.score_mlp = nn.Sequential(
            nn.Linear(d_attn * 4, attn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden_dim, 1),
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor,
        score_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 每个历史位置单独计算与目标的相关性，并可加入侧信息偏置。
        attn_feat = torch.cat([Q, K, Q - K, Q * K], dim=-1)
        score = self.score_mlp(attn_feat).squeeze(-1)
        if self.use_side_bias and score_bias is not None:
            score = score + score_bias
        score = score.masked_fill(mask <= 0, -1e9)

        all_zero_mask = mask.sum(dim=1) <= 0
        alpha = torch.softmax(score, dim=1)
        alpha = alpha * (mask > 0).float()
        alpha_denom = alpha.sum(dim=1, keepdim=True)
        alpha = torch.where(alpha_denom > 0, alpha / alpha_denom, torch.zeros_like(alpha))
        if all_zero_mask.any():
            alpha = alpha.clone()
            alpha[all_zero_mask] = 0.0

        interest = (alpha.unsqueeze(-1) * V).sum(dim=1)
        return interest, alpha, score
