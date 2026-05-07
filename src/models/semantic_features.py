from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoSemanticEncoder(nn.Module):
    def __init__(self, raw_dim: int, semantic_proj_dim: int = 64, semantic_dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, semantic_proj_dim),
            nn.GELU(),
            nn.Dropout(semantic_dropout),
        )

    def forward(self, target_semantic_emb: torch.Tensor) -> torch.Tensor:
        return self.net(target_semantic_emb)


class SimTierEncoder(nn.Module):
    def __init__(self, simtier_input_dim: int, simtier_dim: int = 64, simtier_dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(simtier_input_dim),
            nn.Linear(simtier_input_dim, simtier_dim),
            nn.GELU(),
            nn.Dropout(simtier_dropout),
        )

    def forward(self, simtier_features: torch.Tensor) -> torch.Tensor:
        return self.net(simtier_features)


class SemanticLongShortInterest(nn.Module):
    def __init__(
        self,
        raw_dim: int,
        target_dim: int,
        interest_dim: int = 64,
        gate_hidden_dim: int = 64,
        short_history_len: int = 10,
        history_order: str = "old_to_new",
        simtier_dim: int = 0,
        target_repr_dim: int = 0,
        domain_repr_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.short_history_len = int(short_history_len)
        self.history_order = history_order
        self.hist_proj = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, interest_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.target_proj = nn.Identity() if target_dim == interest_dim else nn.Linear(target_dim, interest_dim)
        gate_input_dim = interest_dim + simtier_dim + target_repr_dim + domain_repr_dim + 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )

    def _masked_cosine_attention(
        self,
        target: torch.Tensor,
        hist: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_n = F.normalize(target, dim=-1).unsqueeze(1)
        hist_n = F.normalize(hist, dim=-1)
        score = (target_n * hist_n).sum(dim=-1)
        score = score.masked_fill(mask <= 0, -1e9)
        alpha = torch.softmax(score, dim=-1) * (mask > 0).to(dtype=hist.dtype)
        denom = alpha.sum(dim=-1, keepdim=True)
        alpha = torch.where(denom > 0, alpha / denom.clamp_min(1e-12), torch.zeros_like(alpha))
        interest = (alpha.unsqueeze(-1) * hist).sum(dim=1)
        return interest, alpha

    def _short_mask(self, hist_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = hist_mask.shape
        k = max(1, min(self.short_history_len, seq_len))
        positions = torch.arange(seq_len, device=hist_mask.device).unsqueeze(0).expand(bsz, -1)
        if self.history_order == "new_to_old":
            mask = positions < k
        else:
            mask = positions >= seq_len - k
        return hist_mask * mask.to(dtype=hist_mask.dtype)

    def forward(
        self,
        target_semantic_repr: torch.Tensor,
        hist_semantic_emb: torch.Tensor,
        hist_mask: torch.Tensor,
        simtier_repr: torch.Tensor | None = None,
        target_repr: torch.Tensor | None = None,
        domain_repr: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        hist_proj = self.hist_proj(hist_semantic_emb)
        target_proj = self.target_proj(target_semantic_repr)
        emb_valid = (hist_semantic_emb.abs().sum(dim=-1) > 0).to(dtype=hist_mask.dtype)
        long_mask = hist_mask.to(dtype=hist_proj.dtype) * emb_valid
        short_mask = self._short_mask(long_mask)
        long_sem_interest, alpha_long = self._masked_cosine_attention(target_proj, hist_proj, long_mask)
        short_sem_interest, alpha_short = self._masked_cosine_attention(target_proj, hist_proj, short_mask)
        hist_len_feature = long_mask.sum(dim=1, keepdim=True).clamp_max(float(hist_mask.size(1))) / max(float(hist_mask.size(1)), 1.0)
        gate_parts = [target_proj, hist_len_feature]
        if simtier_repr is not None:
            gate_parts.insert(1, simtier_repr)
        if target_repr is not None:
            gate_parts.insert(-1, target_repr)
        if domain_repr is not None:
            gate_parts.insert(-1, domain_repr)
        gate = torch.sigmoid(self.gate_mlp(torch.cat(gate_parts, dim=-1)))
        semantic_interest = gate * short_sem_interest + (1.0 - gate) * long_sem_interest
        return {
            "semantic_interest": semantic_interest,
            "short_sem_interest": short_sem_interest,
            "long_sem_interest": long_sem_interest,
            "semantic_gate": gate.squeeze(-1),
            "alpha_short": alpha_short,
            "alpha_long": alpha_long,
            "short_history_non_empty": (short_mask.sum(dim=1) > 0).to(dtype=hist_proj.dtype),
            "long_history_non_empty": (long_mask.sum(dim=1) > 0).to(dtype=hist_proj.dtype),
        }
