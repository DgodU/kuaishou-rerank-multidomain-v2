from __future__ import annotations

import torch
import torch.nn as nn

from src.models.layers import ResidualFFN


# TransformerFusion：对兴趣 token 做二次上下文融合，再根据目标相关性聚合成增强兴趣向量。
class TransformerFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        num_layers: int = 1,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Transformer 编码 token 间关系，适合多兴趣 token 之间的信息交互。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 目标相关注意力决定哪些 token 对当前目标更重要。
        self.target_attn = nn.Sequential(
            nn.Linear(dim * 4, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, 1),
        )
        self.output_ffn = ResidualFFN(dim=dim, hidden_dim=ffn_dim, dropout=dropout)

    def forward(
        self,
        target_emb: torch.Tensor,
        interest_tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 处理全 padding token，避免 Transformer 和 softmax 产生 NaN。
        valid_mask = token_mask > 0
        src_key_padding_mask = ~valid_mask
        all_pad = src_key_padding_mask.all(dim=1)
        encoder_input = interest_tokens
        if all_pad.any():
            src_key_padding_mask = src_key_padding_mask.clone()
            src_key_padding_mask[all_pad, 0] = False
            encoder_input = encoder_input.clone()
            encoder_input[all_pad, 0, :] = 0.0
        z = self.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask)
        if all_pad.any():
            z = z.clone()
            z[all_pad] = 0.0
        target = target_emb.unsqueeze(1).expand_as(z)
        feat = torch.cat([target, z, target - z, target * z], dim=-1)
        score = self.target_attn(feat).squeeze(-1).masked_fill(~valid_mask, -1e9)
        alpha = torch.softmax(score, dim=1) * valid_mask.float()
        denom = alpha.sum(dim=1, keepdim=True)
        alpha = torch.where(denom > 0, alpha / denom, torch.zeros_like(alpha))
        interest = (alpha.unsqueeze(-1) * z).sum(dim=1)
        output = self.output_ffn(interest)
        if all_pad.any():
            output = output.clone()
            output[all_pad] = 0.0
            alpha = alpha.clone()
            alpha[all_pad] = 0.0
        return output, alpha
