from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.layers import PCRG, PSRG, PositionWiseTargetAttention


# ADS 基线模型：构造历史序列 E_S、目标物品 E_Q、场景 E_D，并用目标注意力做 CTR 预测。
class ADSModel(nn.Module):
    def __init__(self, config: Dict, feature_maps: Dict):
        super().__init__()
        self.config = config
        self.feature_maps = feature_maps

        self.embedding_dim = int(config.get("embedding_dim", 32))
        self.max_seq_len = int(config.get("max_seq_len", 50))
        self.d_s = int(config.get("d_s", 64))
        self.d_q = int(config.get("d_q", 64))
        self.d_D = int(config.get("d_D", 64))
        self.d_attn = int(config.get("d_attn", 64))
        self.dropout_p = float(config.get("dropout", 0.1))

        # 基线核心开关：PSRG 个性化历史表示，PCRG 个性化目标查询，category 控制类别特征是否参与。
        self.use_psrg = bool(config.get("use_psrg", True))
        self.use_pcrg = bool(config.get("use_pcrg", True))
        self.use_category = bool(config.get("use_category", True))
        self.ads_generation_mode = str(config.get("ads_generation_mode", "legacy")).strip().lower()

        vocab_sizes = feature_maps["vocab_sizes"]

        def emb(name: str) -> nn.Embedding:
            return nn.Embedding(vocab_sizes[name], self.embedding_dim, padding_idx=0)

        # 基础稀疏特征 Embedding：用户、视频、类别、场景 tab、时长桶等。
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
        # 默认关闭的扩展上下文特征，用于单变量实验和消融。
        self.use_target_tag = bool(config.get("use_target_tag", False))
        self.use_user_profile_context = bool(config.get("use_user_profile_context", False))
        self.use_user_onehot_context = bool(config.get("use_user_onehot_context", False))
        self.register_days_bucket_emb = emb("register_days_bucket")
        self.fans_user_num_bucket_emb = emb("fans_user_num_bucket")
        self.follow_user_num_bucket_emb = emb("follow_user_num_bucket")
        self.friend_user_num_bucket_emb = emb("friend_user_num_bucket")
        self.user_onehot_field_names = []
        self.user_onehot_embs = nn.ModuleDict()
        if self.use_user_onehot_context:
            for i in range(18):
                name = f"onehot_feat{i}"
                if name in vocab_sizes:
                    self.user_onehot_field_names.append(name)
                    self.user_onehot_embs[name] = emb(name)
        self.use_time_context = bool(config.get("use_time_context", False))
        self.hour_emb = None
        self.dow_emb = None
        self.weekend_emb = None
        if self.use_time_context:
            self.hour_emb = nn.Embedding(int(vocab_sizes.get("hour_of_day", 25)), self.embedding_dim, padding_idx=0)
            self.dow_emb = nn.Embedding(int(vocab_sizes.get("day_of_week", 8)), self.embedding_dim, padding_idx=0)
            self.weekend_emb = nn.Embedding(int(vocab_sizes.get("is_weekend", 3)), self.embedding_dim, padding_idx=0)

        # 三路主表示投影：历史序列 E_S、目标物品 E_Q、场景/域 E_D。
        seq_item_fields = 6 if self.use_category else 2
        self.seq_item_proj = nn.Linear(self.embedding_dim * seq_item_fields, self.d_s)
        target_item_fields = 7 if self.use_category else 3
        if self.use_target_tag:
            target_item_fields += 1
        self.target_item_proj = nn.Linear(self.embedding_dim * target_item_fields, self.d_q)

        domain_fields = 6 if self.use_category else 4
        if self.use_time_context:
            domain_fields += 3
        if self.use_user_profile_context:
            domain_fields += 4
        if self.use_user_onehot_context:
            domain_fields += len(self.user_onehot_field_names)
        domain_in = self.embedding_dim * domain_fields
        self.domain_proj = nn.Linear(domain_in, self.d_D)
        generation_domain_dim = domain_in if self.ads_generation_mode == "paper" else self.d_D

        # PSRG/PCRG 根据场景 E_D 对历史表示和目标查询做个性化生成。
        self.psrg = PSRG(
            d_s=self.d_s,
            d_D=generation_domain_dim,
            hidden_dim=int(config.get("psrg_hidden_dim", 128)),
            eta=float(config.get("psrg_eta", 2.0)),
        )
        self.pcrg = PCRG(
            d_q=self.d_q,
            d_D=generation_domain_dim,
            max_seq_len=self.max_seq_len,
            hidden_dim=int(config.get("pcrg_hidden_dim", 128)),
        )

        # 将 Q/K/V 投影到统一注意力空间，用于目标-历史匹配。
        self.linear_k = nn.Linear(self.d_s, self.d_attn)
        self.linear_v = nn.Linear(self.d_s, self.d_attn)
        self.linear_q = nn.Linear(self.d_q, self.d_attn)

        self.attn = PositionWiseTargetAttention(self.d_attn)

        # 最终 MLP 综合用户兴趣、目标物品、用户 ID 和场景表示输出点击 logit。
        final_in = self.d_attn + self.d_q + self.embedding_dim + self.d_D
        self.mlp = nn.Sequential(
            nn.Linear(final_in, 256),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 1),
        )

    def _build_seq_item_emb(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 历史 item 表示：历史视频、类别和时长桶组成 E_S。
        seq_feats = [
            self.video_id_emb(batch["hist_video_id"]),
            self.duration_bucket_emb(batch["hist_duration_bucket"]),
        ]
        if self.use_category:
            seq_feats[1:1] = [
                self.category_l1_emb(batch["hist_category_l1"]),
                self.category_l2_emb(batch["hist_category_l2"]),
                self.category_l3_emb(batch["hist_category_l3"]),
                self.category_l4_emb(batch["hist_category_l4"]),
            ]
        seq_cat = torch.cat(seq_feats, dim=-1)  # [B, T, 6*emb]
        E_S = self.seq_item_proj(seq_cat)  # [B, T, d_s]
        return E_S

    def _build_target_item_emb(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 目标 item 表示：当前待打分视频及其类别、类型、时长等组成 E_Q。
        target_feats = [
            self.video_id_emb(batch["target_video_id"]),
            self.video_type_emb(batch["target_video_type"]),
            self.duration_bucket_emb(batch["target_duration_bucket"]),
        ]
        if self.use_category:
            target_feats[1:1] = [
                self.category_l1_emb(batch["target_category_l1"]),
                self.category_l2_emb(batch["target_category_l2"]),
                self.category_l3_emb(batch["target_category_l3"]),
                self.category_l4_emb(batch["target_category_l4"]),
            ]
        if self.use_target_tag:
            target_feats.append(self.tag_emb(batch["target_tag"]))
        target_cat = torch.cat(target_feats, dim=-1)  # [B, 7*emb]
        E_Q = self.target_item_proj(target_cat)  # [B, d_q]
        return E_Q

    def _build_domain_emb(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 场景/域表示：tab、用户活跃度、目标侧上下文和可选用户/时间特征组成 E_D。
        dom_feats = [
            self.tab_emb(batch["tab"]),
            self.user_active_degree_emb(batch["user_active_degree"]),
            self.video_type_emb(batch["target_video_type"]),
            self.duration_bucket_emb(batch["target_duration_bucket"]),
        ]
        if self.use_category:
            dom_feats[2:2] = [
                self.category_l1_emb(batch["target_category_l1"]),
                self.category_l2_emb(batch["target_category_l2"]),
            ]
        if self.use_user_profile_context:
            dom_feats.extend(
                [
                    self.register_days_bucket_emb(batch["register_days_bucket"]),
                    self.fans_user_num_bucket_emb(batch["fans_user_num_bucket"]),
                    self.follow_user_num_bucket_emb(batch["follow_user_num_bucket"]),
                    self.friend_user_num_bucket_emb(batch["friend_user_num_bucket"]),
                ]
            )
        if self.use_user_onehot_context:
            for name in self.user_onehot_field_names:
                dom_feats.append(self.user_onehot_embs[name](batch[name]))
        if self.use_time_context and self.hour_emb is not None and self.dow_emb is not None and self.weekend_emb is not None:
            dom_feats.extend(
                [
                    self.hour_emb(batch["hour_of_day"]),
                    self.dow_emb(batch["day_of_week"]),
                    self.weekend_emb(batch["is_weekend"]),
                ]
            )
        dom_cat = torch.cat(dom_feats, dim=-1)  # [B, 6*emb]
        E_D = self.domain_proj(dom_cat)  # [B, d_D]
        return E_D, dom_cat

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hist_mask = batch["hist_mask"]  # [B, T]

        # 1) E_S: [B, T, d_s]
        E_S = self._build_seq_item_emb(batch)
        # 2) E_Q: [B, d_q]
        E_Q = self._build_target_item_emb(batch)
        # 3) E_D: [B, d_D]
        E_D, E_D_raw = self._build_domain_emb(batch)
        E_D_generation = E_D_raw if self.ads_generation_mode == "paper" else E_D

        # 4) PSRG：用 E_D 调制历史序列表示，使历史兴趣具有场景个性化。
        if self.use_psrg:
            E_S_personalized = self.psrg(E_S, E_D_generation)  # [B, T, d_s]
        else:
            E_S_personalized = E_S
        E_S_personalized = E_S_personalized * hist_mask.unsqueeze(-1)

        # 5) PCRG：用 E_Q 和 E_D 为每个历史位置生成个性化 query。
        if self.use_pcrg:
            Q_personalized = self.pcrg(E_Q, E_D_generation)  # [B, T, d_q]
        else:
            Q_personalized = E_Q.unsqueeze(1).expand(-1, self.max_seq_len, -1)

        # 6/7/8) project to attn space
        K = self.linear_k(E_S_personalized)  # [B, T, d_attn]
        V = self.linear_v(E_S_personalized)  # [B, T, d_attn]
        Q = self.linear_q(Q_personalized)  # [B, T, d_attn]

        # 9) Position-wise target attention：按目标相关性聚合历史兴趣。
        interest, alpha = self.attn(Q, K, V, hist_mask)  # interest: [B, d_attn], alpha: [B, T]

        # 10) final input：拼接兴趣、目标、用户和场景信息。
        user_emb = self.user_id_emb(batch["user_id"])  # [B, emb]
        final_input = torch.cat([interest, E_Q, user_emb, E_D], dim=-1)  # [B, d_attn+d_q+emb+d_D]

        # 11) logit
        logit = self.mlp(final_input).squeeze(-1)  # [B]
        pred = torch.sigmoid(logit)

        return {"logit": logit, "pred": pred, "attn": alpha}
