from __future__ import annotations

import math

import torch
import torch.nn as nn


# PSRG：根据场景/域表示 E_D 动态生成历史序列变换参数，用于得到个性化历史行为表示。
class PSRG(nn.Module):
    def __init__(self, d_s: int, d_D: int, hidden_dim: int = 128, eta: float = 2.0):
        super().__init__()
        self.d_s = d_s
        self.eta = eta

        # 由 E_D 生成每个样本自己的私有权重和偏置。
        self.mlp_weight = nn.Sequential(
            nn.Linear(d_D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_s * d_s),
        )
        self.mlp_bias = nn.Sequential(
            nn.Linear(d_D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_s),
        )

        # 共享权重提供全局基础变换，私有权重负责按样本调制。
        self.W_shared = nn.Parameter(torch.empty(d_s, d_s))
        nn.init.xavier_uniform_(self.W_shared)

    def forward(
        self,
        seq_emb: torch.Tensor,
        domain_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # seq_emb: [B, T, d_s], domain_emb: [B, d_D]
        B = seq_emb.size(0)

        # 生成样本级矩阵，并与共享矩阵逐元素相乘形成最终变换。
        W_private = torch.sigmoid(self.mlp_weight(domain_emb))  # [B, d_s*d_s]
        W_private = W_private.view(B, self.d_s, self.d_s)  # [B, d_s, d_s]

        W_generated = self.eta * (self.W_shared.unsqueeze(0) * W_private)  # [B, d_s, d_s]
        b_generated = self.mlp_bias(domain_emb)  # [B, d_s]

        out = torch.bmm(seq_emb, W_generated.transpose(1, 2)) + b_generated.unsqueeze(1)
        # out: [B, T, d_s]
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


# PCRG：根据目标物品 E_Q 和场景 E_D，为每个历史位置生成个性化查询向量。
class PCRG(nn.Module):
    def __init__(self, d_q: int, d_D: int, max_seq_len: int, hidden_dim: int = 128):
        super().__init__()
        self.d_q = d_q
        self.max_seq_len = max_seq_len
        # 一次性生成 T 个位置的 query，再 reshape 成序列查询。
        self.mlp_query = nn.Sequential(
            nn.Linear(d_q + d_D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_seq_len * d_q),
        )

    def forward(self, target_emb: torch.Tensor, domain_emb: torch.Tensor) -> torch.Tensor:
        # target_emb: [B, d_q], domain_emb: [B, d_D]
        B = target_emb.size(0)
        x = torch.cat([domain_emb, target_emb], dim=-1)  # [B, d_D+d_q]

        q_private_flat = self.mlp_query(x)  # [B, T*d_q]
        q_private = q_private_flat.view(B, self.max_seq_len, self.d_q)  # [B, T, d_q]

        # 私有查询叠加共享目标查询，兼顾个性化与目标自身语义。
        q_shared = target_emb.unsqueeze(1).expand(-1, self.max_seq_len, -1)  # [B, T, d_q]
        q_personalized = q_private + q_shared  # [B, T, d_q]
        return q_personalized


# 逐位置目标注意力：计算目标查询和每个历史位置的匹配分数，聚合成用户兴趣向量。
class PositionWiseTargetAttention(nn.Module):
    def __init__(self, d_attn: int):
        super().__init__()
        self.scale = math.sqrt(float(d_attn))

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor,
        score_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Q/K/V: [B, T, d_attn], mask: [B, T]
        # 点积匹配分数，可叠加 Side Attention Bias 等外部偏置。
        score = (Q * K).sum(dim=-1) / self.scale  # [B, T]
        if score_bias is not None:
            score = score + score_bias
        score = score.masked_fill(mask <= 0, -1e9)

        # mask 负责屏蔽 padding；全空历史样本的注意力输出置零，避免 NaN。
        all_zero_mask = mask.sum(dim=1) <= 0  # [B]
        alpha = torch.softmax(score, dim=1)  # [B, T]
        alpha = alpha * (mask > 0).float()

        alpha_denom = alpha.sum(dim=1, keepdim=True)  # [B, 1]
        alpha = torch.where(alpha_denom > 0, alpha / alpha_denom, torch.zeros_like(alpha))

        if all_zero_mask.any():
            alpha[all_zero_mask] = 0.0

        interest = (alpha.unsqueeze(-1) * V).sum(dim=1)  # [B, d_attn]
        return interest, alpha


# 残差前馈层：增强兴趣向量表达，同时保持输入输出维度一致。
class ResidualFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim]
        out = self.fc2(self.dropout(self.act(self.fc1(x))))
        out = self.norm(x + self.dropout(out))
        return out
