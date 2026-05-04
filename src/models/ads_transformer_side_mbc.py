from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import PCRG, PSRG, PositionWiseTargetAttention, ResidualFFN
from src.models.static_mbc import StaticMBCInteractionLayer


# ADS-Transformer-SideInfo-MBC：在序列兴趣建模之外，引入静态字段 MBC 多分支交互做 CTR 预测。
class ADSTransformerSideMBCModel(nn.Module):
    def __init__(self, config: Dict, feature_maps: Dict):
        super().__init__()
        self.config = config
        self.feature_maps = feature_maps

        self.max_seq_len = int(config.get("max_seq_len", 50))
        self.field_embedding_dim = int(config.get("field_embedding_dim", 16))
        self.sequence_item_dim = int(config.get("sequence_item_dim", 64))
        self.target_item_dim = int(config.get("target_item_dim", 64))
        self.domain_dim = int(config.get("domain_dim", 64))
        self.d_attn = int(config.get("d_attn", 64))
        self.dropout = float(config.get("dropout", 0.1))

        # 主功能开关：控制个性化生成、Transformer 序列建模、行为侧信息和静态 MBC 分支。
        self.use_psrg = bool(config.get("use_psrg", True))
        self.use_pcrg = bool(config.get("use_pcrg", True))
        self.use_transformer = bool(config.get("use_transformer", True))
        self.use_behavior_side = bool(config.get("use_behavior_side", True))
        self.use_mbc = bool(config.get("use_mbc", True)) and bool(config.get("use_static_mbc", True))

        # 可选辅助损失：训练器可读取各分支输出，用于分支监督或多样性约束实验。
        self.use_mbc_aux_loss = bool(config.get("use_mbc_aux_loss", False))
        self.use_mbc_diversity_loss = bool(config.get("use_mbc_diversity_loss", False))

        self.use_user_onehot_feats = bool(config.get("use_user_onehot_feats", False))

        self.debug_shapes = bool(config.get("_debug_shapes", False))
        self._shape_logged = False

        vocab_sizes = feature_maps["vocab_sizes"]
        bucket_sizes = feature_maps.get("bucket_sizes", {})

        def emb(name: str) -> nn.Embedding:
            if name not in vocab_sizes:
                raise KeyError(f"Missing vocab size for '{name}' in feature_maps")
            return nn.Embedding(int(vocab_sizes[name]), self.field_embedding_dim, padding_idx=0)

        # 静态和序列字段共享 Embedding，后续分别组成历史、目标、场景和 MBC 静态字段。
        self.user_id_emb = emb("user_id")
        self.video_id_emb = emb("video_id")
        self.tab_emb = emb("tab")
        self.user_active_degree_emb = emb("user_active_degree")
        self.category_l1_emb = emb("category_l1_id")
        self.category_l2_emb = emb("category_l2_id")
        self.category_l3_emb = emb("category_l3_id")
        self.category_l4_emb = emb("category_l4_id")
        self.video_type_emb = emb("video_type")
        self.duration_bucket_emb = emb("duration_bucket")
        self.tag_emb = emb("tag")

        self.register_days_bucket_emb = emb("register_days_bucket")
        self.fans_user_num_bucket_emb = emb("fans_user_num_bucket")
        self.follow_user_num_bucket_emb = emb("follow_user_num_bucket")
        self.friend_user_num_bucket_emb = emb("friend_user_num_bucket")

        self.onehot_embs = nn.ModuleDict()
        if self.use_user_onehot_feats:
            for i in range(18):
                name = f"onehot_feat{i}"
                if name in vocab_sizes:
                    self.onehot_embs[name] = emb(name)

        # 三路投影：历史 E_S、目标 E_Q、场景 E_D。
        self.hist_item_proj = nn.Linear(self.field_embedding_dim * 6, self.sequence_item_dim)
        self.target_item_proj = nn.Linear(self.field_embedding_dim * 8, self.target_item_dim)
        self.domain_proj = nn.Linear(self.field_embedding_dim * 6, self.domain_dim)

        # ADS 主干中的 PSRG/PCRG：分别个性化历史序列表示和目标查询。
        self.psrg = PSRG(
            d_s=self.sequence_item_dim,
            d_D=self.domain_dim,
            hidden_dim=int(config.get("psrg_hidden_dim", 128)),
            eta=float(config.get("psrg_eta", 2.0)),
        )
        self.pcrg = PCRG(
            d_q=self.target_item_dim,
            d_D=self.domain_dim,
            max_seq_len=self.max_seq_len,
            hidden_dim=int(config.get("pcrg_hidden_dim", 128)),
        )

        # 行为侧信息：动作向量、播放比例、时间间隔、历史 tab 和位置编码。
        side_dim = int(config.get("side_dim", 16))
        side_output_dim = int(config.get("side_output_dim", 32))
        self.action_linear = nn.Linear(7, side_dim)
        self.play_ratio_emb = nn.Embedding(
            int(bucket_sizes.get("play_ratio_bucket", 1)) + 1,
            side_dim,
            padding_idx=0,
        )
        self.time_gap_emb = nn.Embedding(
            int(bucket_sizes.get("time_gap_bucket", 1)) + 1,
            side_dim,
            padding_idx=0,
        )
        self.hist_tab_emb = nn.Embedding(vocab_sizes["tab"], side_dim, padding_idx=0)
        self.position_emb = nn.Embedding(self.max_seq_len + 1, side_dim, padding_idx=0)

        self.side_proj = nn.Linear(side_dim * 5, side_output_dim)

        d_model = int(config.get("transformer_d_model", 64))
        self.h_proj = nn.Linear(self.sequence_item_dim, d_model)
        self.side_to_model = nn.Linear(side_output_dim, d_model)
        self.side_gate = nn.Linear(d_model, d_model)
        self.side_post_norm = nn.LayerNorm(d_model)

        # Transformer 编码历史序列上下文，随后用门控方式注入 SideInfo。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config.get("transformer_heads", 4)),
            dim_feedforward=int(config.get("transformer_ffn_dim", 256)),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.get("transformer_layers", 1)),
        )

        self.linear_k = nn.Linear(d_model, self.d_attn)
        self.linear_v = nn.Linear(d_model, self.d_attn)
        self.linear_q = nn.Linear(self.target_item_dim, self.d_attn)
        self.attn = PositionWiseTargetAttention(self.d_attn)

        self.interest_ffn = ResidualFFN(
            dim=self.d_attn,
            hidden_dim=int(config.get("interest_ffn_dim", 256)),
            dropout=self.dropout,
        )

        # 静态 MBC 使用用户、目标视频、类别、场景等字段建模跨字段交互。
        self.static_field_names: List[str] = [
            "user_id",
            "user_active_degree",
            "register_days_bucket",
            "fans_user_num_bucket",
            "follow_user_num_bucket",
            "friend_user_num_bucket",
            "target_video_id",
            "target_category_l1",
            "target_category_l2",
            "target_category_l3",
            "target_category_l4",
            "target_video_type",
            "target_duration_bucket",
            "target_tag",
            "tab",
        ]
        if self.use_user_onehot_feats:
            for i in range(18):
                name = f"onehot_feat{i}"
                if name in self.onehot_embs:
                    self.static_field_names.append(name)

        # MBC 多分支静态交互模块，输出与序列兴趣向量融合。
        self.static_mbc = StaticMBCInteractionLayer(
            field_embedding_dim=self.field_embedding_dim,
            static_field_names=self.static_field_names,
            config=config,
        )

        fusion_dim = int(config.get("mbc_fusion_dim", 64))
        if str(config.get("mbc_fusion_type", "mean")).lower() == "concat":
            enabled_branches = sum(
                [
                    bool(config.get("use_efgc_branch", True)),
                    bool(config.get("use_cross_branch", True)),
                    bool(config.get("use_deep_branch", True)),
                ]
            )
            fusion_dim = fusion_dim * max(enabled_branches, 1)

        # 最终 MLP 输入为序列兴趣向量 + 静态 MBC 融合向量。
        final_hidden_dims = list(config.get("final_hidden_dims", [128, 64]))
        final_input_dim = self.d_attn + fusion_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(final_input_dim, int(final_hidden_dims[0])),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(final_hidden_dims[0]), int(final_hidden_dims[1])),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(final_hidden_dims[1]), 1),
        )

        # 静态融合门控控制 MBC 静态交互对最终预测的贡献强度。
        self.use_static_fusion_gate = bool(config.get("use_static_fusion_gate", True))
        static_gate_hidden_dim = int(config.get("static_gate_hidden_dim", 64))
        self.static_fusion_gate_mlp = nn.Sequential(
            nn.Linear(self.d_attn + fusion_dim, static_gate_hidden_dim),
            nn.GELU(),
            nn.Linear(static_gate_hidden_dim, 1),
        )

        # 分支 head 为辅助损失提供各 MBC 分支的独立 logit。
        self.branch_head_efgc = nn.Linear(int(config.get("mbc_fusion_dim", 64)), 1)
        self.branch_head_cross = nn.Linear(int(config.get("mbc_fusion_dim", 64)), 1)
        self.branch_head_deep = nn.Linear(int(config.get("mbc_fusion_dim", 64)), 1)
        self.branch_head_fusion = nn.Linear(int(config.get("mbc_fusion_dim", 64)), 1)

    def _build_hist_item_emb(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 历史 item 表示：历史视频及其多级类别、时长桶组成 E_S。
        hist_fields = [
            self.video_id_emb(batch["hist_video_id"]),
            self.category_l1_emb(batch["hist_category_l1"]),
            self.category_l2_emb(batch["hist_category_l2"]),
            self.category_l3_emb(batch["hist_category_l3"]),
            self.category_l4_emb(batch["hist_category_l4"]),
            self.duration_bucket_emb(batch["hist_duration_bucket"]),
        ]
        hist_cat = torch.cat(hist_fields, dim=-1)  # [B, T, 6*field_emb]
        E_S = self.hist_item_proj(hist_cat)  # [B, T, sequence_item_dim]
        return E_S

    def _build_target_item_emb(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 目标 item 表示：返回 E_Q，同时保留字段级 embedding 给静态 MBC 使用。
        target_field_embs = {
            "target_video_id": self.video_id_emb(batch["target_video_id"]),
            "target_category_l1": self.category_l1_emb(batch["target_category_l1"]),
            "target_category_l2": self.category_l2_emb(batch["target_category_l2"]),
            "target_category_l3": self.category_l3_emb(batch["target_category_l3"]),
            "target_category_l4": self.category_l4_emb(batch["target_category_l4"]),
            "target_video_type": self.video_type_emb(batch["target_video_type"]),
            "target_duration_bucket": self.duration_bucket_emb(batch["target_duration_bucket"]),
            "target_tag": self.tag_emb(batch["target_tag"]),
        }
        target_cat = torch.cat(list(target_field_embs.values()), dim=-1)  # [B, 8*field_emb]
        E_Q = self.target_item_proj(target_cat)  # [B, target_item_dim]
        return E_Q, target_field_embs

    def _build_domain_emb(self, batch: Dict[str, torch.Tensor], target_field_embs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 场景/域表示：tab、用户活跃度和目标侧上下文组成 E_D。
        domain_fields = [
            self.tab_emb(batch["tab"]),
            self.user_active_degree_emb(batch["user_active_degree"]),
            target_field_embs["target_category_l1"],
            target_field_embs["target_category_l2"],
            target_field_embs["target_video_type"],
            target_field_embs["target_duration_bucket"],
        ]
        domain_cat = torch.cat(domain_fields, dim=-1)  # [B, 6*field_emb]
        E_D = self.domain_proj(domain_cat)  # [B, domain_dim]
        return E_D

    def _build_side_info(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 行为侧信息描述每个历史位置的行为强度、时间间隔、来源 tab 和位置。
        hist_action = batch["hist_action_vector"]  # [B, T, 7]
        hist_play_ratio_bucket = batch["hist_play_ratio_bucket"]  # [B, T]
        hist_time_gap_bucket = batch["hist_time_gap_bucket"]  # [B, T]
        hist_tab = batch["hist_tab"]  # [B, T]
        hist_mask = batch["hist_mask"]  # [B, T]

        B, T = hist_tab.shape
        pos = torch.arange(1, T + 1, device=hist_tab.device).unsqueeze(0).expand(B, -1)  # [B, T]
        pos = pos * (hist_mask > 0).long()

        action_emb = self.action_linear(hist_action)  # [B, T, side_dim]
        play_emb = self.play_ratio_emb(hist_play_ratio_bucket)  # [B, T, side_dim]
        gap_emb = self.time_gap_emb(hist_time_gap_bucket)  # [B, T, side_dim]
        tab_emb = self.hist_tab_emb(hist_tab)  # [B, T, side_dim]
        pos_emb = self.position_emb(pos)  # [B, T, side_dim]

        side_concat = torch.cat([action_emb, play_emb, gap_emb, tab_emb, pos_emb], dim=-1)
        # side_concat: [B, T, 5*side_dim]
        side_info = self.side_proj(side_concat)  # [B, T, side_output_dim]
        return side_info

    def _build_static_field_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
        target_field_embs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # 构造静态字段 embedding 字典，供 EFGC/Cross/Deep 三类 MBC 分支复用。
        static_fields: Dict[str, torch.Tensor] = {
            "user_id": self.user_id_emb(batch["user_id"]),
            "user_active_degree": self.user_active_degree_emb(batch["user_active_degree"]),
            "register_days_bucket": self.register_days_bucket_emb(batch["register_days_bucket"]),
            "fans_user_num_bucket": self.fans_user_num_bucket_emb(batch["fans_user_num_bucket"]),
            "follow_user_num_bucket": self.follow_user_num_bucket_emb(batch["follow_user_num_bucket"]),
            "friend_user_num_bucket": self.friend_user_num_bucket_emb(batch["friend_user_num_bucket"]),
            "target_video_id": target_field_embs["target_video_id"],
            "target_category_l1": target_field_embs["target_category_l1"],
            "target_category_l2": target_field_embs["target_category_l2"],
            "target_category_l3": target_field_embs["target_category_l3"],
            "target_category_l4": target_field_embs["target_category_l4"],
            "target_video_type": target_field_embs["target_video_type"],
            "target_duration_bucket": target_field_embs["target_duration_bucket"],
            "target_tag": target_field_embs["target_tag"],
            "tab": self.tab_emb(batch["tab"]),
        }

        if self.use_user_onehot_feats:
            for i in range(18):
                name = f"onehot_feat{i}"
                if name in self.onehot_embs and name in batch:
                    static_fields[name] = self.onehot_embs[name](batch[name])

        return static_fields

    def _maybe_log_shapes(self, shape_map: Dict[str, torch.Size]) -> None:
        if not self.debug_shapes or self._shape_logged:
            return
        for k, v in shape_map.items():
            print(f"{k} shape: {tuple(v)}")
        self._shape_logged = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hist_mask = batch["hist_mask"]  # [B, T]

        # 1) E_S: [B, T, sequence_item_dim]
        E_S = self._build_hist_item_emb(batch)

        # 2) E_Q: [B, target_item_dim]
        E_Q, target_field_embs = self._build_target_item_emb(batch)

        # 3) E_D: [B, domain_dim]
        E_D = self._build_domain_emb(batch, target_field_embs)

        # 4) PSRG：用场景表示个性化历史序列。
        if self.use_psrg:
            E_S_personalized = self.psrg(E_S, E_D, mask=hist_mask)
        else:
            E_S_personalized = E_S * hist_mask.unsqueeze(-1)

        # 5/6) Transformer：仅建模历史序列上下文。
        H_proj = self.h_proj(E_S_personalized)  # [B, T, transformer_d_model]

        if self.use_transformer:
            src_key_padding_mask = hist_mask <= 0  # [B, T]
            all_pad = src_key_padding_mask.all(dim=1)  # [B]
            if all_pad.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_pad, 0] = False
                H_proj = H_proj.clone()
                H_proj[all_pad, 0, :] = 0.0

            H_ctx = self.transformer_encoder(H_proj, src_key_padding_mask=src_key_padding_mask)
            if all_pad.any():
                H_ctx = H_ctx.clone()
                H_ctx[all_pad] = 0.0
        else:
            H_ctx = H_proj
        H_ctx = H_ctx * hist_mask.unsqueeze(-1)  # [B, T, transformer_d_model]

        # 7) SideInfo post-Transformer residual gated injection：门控注入行为侧信息。
        if self.use_behavior_side:
            side_info = self._build_side_info(batch)  # [B, T, side_output_dim]
        else:
            side_info = torch.zeros(
                E_S_personalized.size(0),
                E_S_personalized.size(1),
                self.side_proj.out_features,
                dtype=E_S_personalized.dtype,
                device=E_S_personalized.device,
            )

        side_emb = self.side_to_model(side_info)  # [B, T, d_model]
        gate = torch.sigmoid(self.side_gate(side_emb)) * hist_mask.unsqueeze(-1)  # [B, T, d_model]
        H_with_side = self.side_post_norm(H_ctx + gate * side_emb) * hist_mask.unsqueeze(-1)  # [B, T, d_model]

        # 8) K/V：由注入 SideInfo 后的历史上下文生成注意力 key/value。
        K = self.linear_k(H_with_side)  # [B, T, d_attn]
        V = self.linear_v(H_with_side)  # [B, T, d_attn]

        # 8) PCRG + Q：目标和场景生成位置相关 query。
        if self.use_pcrg:
            Q_personalized = self.pcrg(E_Q, E_D)  # [B, T, target_item_dim]
        else:
            Q_personalized = E_Q.unsqueeze(1).expand(-1, self.max_seq_len, -1)
        Q = self.linear_q(Q_personalized)  # [B, T, d_attn]

        # 9) attention
        interest, alpha = self.attn(Q, K, V, hist_mask)  # interest: [B, d_attn], alpha: [B, T]

        # 10) interest ffn
        interest_enhanced = self.interest_ffn(interest)  # [B, d_attn]

        # 11/12) static MBC：静态字段多分支交互，补充序列兴趣以外的先验。
        static_field_embeddings = self._build_static_field_embeddings(batch, target_field_embs)
        if self.use_mbc:
            static_out = self.static_mbc(static_field_embeddings)
            static_mbc_vector = static_out["fusion"]
        else:
            B = interest_enhanced.size(0)
            static_mbc_vector = torch.zeros(B, int(self.config.get("mbc_fusion_dim", 64)), device=interest_enhanced.device)
            static_out = {"efgc": None, "cross": None, "deep": None}

        static_gate = torch.ones(
            static_mbc_vector.size(0),
            1,
            dtype=static_mbc_vector.dtype,
            device=static_mbc_vector.device,
        )
        if self.use_static_fusion_gate:
            gate_in = torch.cat([interest_enhanced, static_mbc_vector], dim=-1)
            static_gate = torch.sigmoid(self.static_fusion_gate_mlp(gate_in))
            static_mbc_vector = static_mbc_vector * static_gate

        # 13) final input：序列兴趣与静态 MBC 向量融合。
        final_input = torch.cat([interest_enhanced, static_mbc_vector], dim=-1)
        # final_input: [B, d_attn + mbc_fusion_dim(or concat fusion dim)]

        # 14) final logit
        logit = self.final_mlp(final_input).squeeze(-1)  # [B]
        pred = torch.sigmoid(logit)

        branch_logits = {}
        branch_vectors = {
            "efgc": static_out.get("efgc"),
            "cross": static_out.get("cross"),
            "deep": static_out.get("deep"),
            "static_fusion": static_mbc_vector,
            "static_gate": static_gate,
        }

        if static_out.get("efgc") is not None:
            branch_logits["efgc"] = self.branch_head_efgc(static_out["efgc"]).squeeze(-1)
        if static_out.get("cross") is not None:
            branch_logits["cross"] = self.branch_head_cross(static_out["cross"]).squeeze(-1)
        if static_out.get("deep") is not None:
            branch_logits["deep"] = self.branch_head_deep(static_out["deep"]).squeeze(-1)
        if static_out.get("fusion") is not None and static_out["fusion"].shape[-1] == self.branch_head_fusion.in_features:
            branch_logits["static_fusion"] = self.branch_head_fusion(static_out["fusion"]).squeeze(-1)

        self._maybe_log_shapes(
            {
                "E_S": E_S.shape,
                "E_Q": E_Q.shape,
                "E_D": E_D.shape,
                "E_S_personalized": E_S_personalized.shape,
                "H_ctx": H_ctx.shape,
                "Q/K/V": (Q.shape, K.shape, V.shape),
                "Q": Q.shape,
                "K": K.shape,
                "V": V.shape,
                "interest_enhanced": interest_enhanced.shape,
                "static_mbc_vector": static_mbc_vector.shape,
                "final_input": final_input.shape,
                "logit": logit.shape,
            }
        )

        return {
            "logit": logit,
            "pred": pred,
            "attn": alpha,
            "interest": interest_enhanced,
            "static_mbc_vector": static_mbc_vector,
            "branch_logits": branch_logits,
            "branch_vectors": branch_vectors,
        }
