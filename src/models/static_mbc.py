from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# 通用 MLP 块：供 MBC 深层分支复用。
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.extend([nn.Linear(prev, out_dim), nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        return self.net(x)


# EFGC 分支：按预定义字段组建模静态特征交互。
class EFGCBranch(nn.Module):
    def __init__(
        self,
        field_embedding_dim: int,
        group_defs: Dict[str, List[str]],
        group_hidden_dim: int,
        group_output_dim: int,
        branch_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.group_defs = group_defs
        self.group_mlps = nn.ModuleDict()

        # 每个字段组独立建模，再把组输出拼接成分支向量。
        for g_name, g_fields in self.group_defs.items():
            in_dim = len(g_fields) * field_embedding_dim
            self.group_mlps[g_name] = nn.Sequential(
                nn.Linear(in_dim, group_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(group_hidden_dim, group_output_dim),
                nn.GELU(),
            )

        self.output_proj = nn.Sequential(
            nn.Linear(len(self.group_defs) * group_output_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, static_field_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        group_outputs = []
        for g_name, g_fields in self.group_defs.items():
            missing = [f for f in g_fields if f not in static_field_embeddings]
            if missing:
                raise KeyError(f"EFGC group '{g_name}' missing fields: {missing}")

            g_cat = torch.cat([static_field_embeddings[f] for f in g_fields], dim=-1)
            # g_cat: [B, len(group_fields)*field_embedding_dim]
            g_out = self.group_mlps[g_name](g_cat)
            # g_out: [B, group_output_dim]
            group_outputs.append(g_out)

        h_groups = torch.cat(group_outputs, dim=-1)
        # h_groups: [B, num_groups*group_output_dim]
        h_efgc = self.output_proj(h_groups)
        # h_efgc: [B, mbc_branch_dim]
        return h_efgc


# 低秩 Cross 层：用低秩映射近似显式交叉，降低高维静态特征交互成本。
class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, rank: int, dropout: float):
        super().__init__()
        self.v = nn.Linear(input_dim, rank, bias=False)
        self.u = nn.Linear(rank, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0/xl: [B, input_dim]
        v = F.gelu(self.v(xl))  # [B, rank]
        u = self.dropout(self.u(v))  # [B, input_dim]
        x_next = x0 * u + xl  # [B, input_dim]
        return x_next


# 低秩 CrossNet 分支：多层低秩交叉后输出一个 MBC 分支向量。
class LowRankCrossNet(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, rank: int, branch_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [LowRankCrossLayer(input_dim=input_dim, rank=rank, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        h_cross = self.output_proj(xl)
        # h_cross: [B, mbc_branch_dim]
        return h_cross


# DeepNet 分支：直接用 MLP 学习拼接静态字段的高阶非线性交互。
class DeepNetBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], branch_dim: int, dropout: float):
        super().__init__()
        self.mlp = MLPBlock(input_dim, hidden_dims, branch_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        h_deep = self.mlp(x)
        # h_deep: [B, mbc_branch_dim]
        return h_deep


# 共享顶层：把不同 MBC 分支统一投影到相同 fusion 维度。
class SharedTopLayer(nn.Module):
    def __init__(self, branch_dim: int, hidden_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, mbc_branch_dim]
        z = self.net(x)
        # z: [B, mbc_fusion_dim]
        return z


# 静态 MBC 交互层：融合 EFGC、CrossNet、DeepNet 等静态字段交互分支。
class StaticMBCInteractionLayer(nn.Module):
    def __init__(
        self,
        field_embedding_dim: int,
        static_field_names: List[str],
        config: Dict,
    ):
        super().__init__()
        self.static_field_names = static_field_names
        self.field_embedding_dim = field_embedding_dim

        self.mbc_branch_dim = int(config.get("mbc_branch_dim", 128))
        self.mbc_fusion_dim = int(config.get("mbc_fusion_dim", 64))
        self.mbc_fusion_type = str(config.get("mbc_fusion_type", "mean")).lower()
        self.dropout = float(config.get("dropout", 0.1))

        self.use_efgc_branch = bool(config.get("use_efgc_branch", True))
        self.use_cross_branch = bool(config.get("use_cross_branch", True))
        self.use_deep_branch = bool(config.get("use_deep_branch", True))

        if not (self.use_efgc_branch or self.use_cross_branch or self.use_deep_branch):
            raise ValueError("At least one MBC branch must be enabled.")

        # 字段组定义体现不同业务语义：用户-目标偏好、目标-场景匹配、类别语义等。
        self.group_defs = {
            "user_target_preference": [
                "user_id",
                "target_video_id",
                "target_category_l1",
                "target_category_l2",
                "target_tag",
            ],
            "user_domain_preference": [
                "user_id",
                "user_active_degree",
                "tab",
                "target_category_l1",
            ],
            "target_domain_matching": [
                "target_video_id",
                "target_video_type",
                "target_duration_bucket",
                "tab",
            ],
            "user_activity_content": [
                "user_active_degree",
                "register_days_bucket",
                "fans_user_num_bucket",
                "target_video_type",
                "target_duration_bucket",
            ],
            "category_semantics": [
                "target_category_l1",
                "target_category_l2",
                "target_category_l3",
                "target_category_l4",
                "target_tag",
            ],
        }

        input_dim = len(self.static_field_names) * self.field_embedding_dim

        # 三个分支可单独开关，便于做消融实验。
        if self.use_efgc_branch:
            self.efgc_branch = EFGCBranch(
                field_embedding_dim=self.field_embedding_dim,
                group_defs=self.group_defs,
                group_hidden_dim=int(config.get("efgc_group_hidden_dim", 64)),
                group_output_dim=int(config.get("efgc_group_output_dim", 32)),
                branch_dim=self.mbc_branch_dim,
                dropout=self.dropout,
            )

        if self.use_cross_branch:
            self.cross_branch = LowRankCrossNet(
                input_dim=input_dim,
                num_layers=int(config.get("cross_layers", 2)),
                rank=int(config.get("cross_low_rank", 16)),
                branch_dim=self.mbc_branch_dim,
                dropout=self.dropout,
            )

        if self.use_deep_branch:
            self.deep_branch = DeepNetBranch(
                input_dim=input_dim,
                hidden_dims=list(config.get("deep_hidden_dims", [256, 128])),
                branch_dim=self.mbc_branch_dim,
                dropout=self.dropout,
            )

        # 所有启用分支共享顶层，减少参数并对齐输出空间。
        self.shared_top = SharedTopLayer(
            branch_dim=self.mbc_branch_dim,
            hidden_dim=int(config.get("shared_top_hidden_dim", 128)),
            fusion_dim=self.mbc_fusion_dim,
            dropout=self.dropout,
        )

    def forward(self, static_field_embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        missing = [f for f in self.static_field_names if f not in static_field_embeddings]
        if missing:
            raise KeyError(f"Missing static fields for MBC: {missing}")

        x_static = torch.cat([static_field_embeddings[name] for name in self.static_field_names], dim=-1)
        # x_static: [B, num_static_fields*field_embedding_dim]

        branch_outputs: Dict[str, torch.Tensor] = {}
        z_list = []

        if self.use_efgc_branch:
            h_efgc = self.efgc_branch(static_field_embeddings)  # [B, mbc_branch_dim]
            z_efgc = self.shared_top(h_efgc)  # [B, mbc_fusion_dim]
            branch_outputs["efgc"] = z_efgc
            z_list.append(z_efgc)

        if self.use_cross_branch:
            h_cross = self.cross_branch(x_static)  # [B, mbc_branch_dim]
            z_cross = self.shared_top(h_cross)  # [B, mbc_fusion_dim]
            branch_outputs["cross"] = z_cross
            z_list.append(z_cross)

        if self.use_deep_branch:
            h_deep = self.deep_branch(x_static)  # [B, mbc_branch_dim]
            z_deep = self.shared_top(h_deep)  # [B, mbc_fusion_dim]
            branch_outputs["deep"] = z_deep
            z_list.append(z_deep)

        if self.mbc_fusion_type == "concat":
            z_static_fusion = torch.cat(z_list, dim=-1)
            # z_static_fusion: [B, num_enabled*mbc_fusion_dim]
        else:
            z_stack = torch.stack(z_list, dim=1)  # [B, num_enabled, mbc_fusion_dim]
            z_static_fusion = z_stack.mean(dim=1)  # [B, mbc_fusion_dim]

        out = {
            "fusion": z_static_fusion,
            "efgc": branch_outputs.get("efgc"),
            "cross": branch_outputs.get("cross"),
            "deep": branch_outputs.get("deep"),
        }
        return out
