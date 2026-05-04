from __future__ import annotations

import torch
import torch.nn as nn


# PCRG Token 层：由目标物品和场景生成多个兴趣查询 token，用于多兴趣注意力实验。
class PCRGTokenLayer(nn.Module):
    def __init__(self, target_dim: int, domain_dim: int, num_interest_tokens: int = 4, token_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.num_interest_tokens = num_interest_tokens
        self.token_dim = token_dim
        # 输出 num_interest_tokens 个 token，每个 token 表示一种可能的用户兴趣查询。
        self.mlp = nn.Sequential(
            nn.Linear(target_dim + domain_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_interest_tokens * token_dim),
        )

    def forward(self, target_emb: torch.Tensor, domain_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 目标语义和场景语义共同决定本样本的兴趣查询 token。
        x = torch.cat([target_emb, domain_emb], dim=-1)
        q_flat = self.mlp(x)
        queries = q_flat.view(target_emb.size(0), self.num_interest_tokens, self.token_dim)
        token_mask = torch.ones(
            target_emb.size(0),
            self.num_interest_tokens,
            device=target_emb.device,
            dtype=target_emb.dtype,
        )
        return queries, token_mask
