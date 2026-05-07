from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import math

from src.models.ads import ADSModel
from src.models.attention import DINStyleTargetAttention
from src.models.calibration_head import CalibrationHead
from src.models.layers import ResidualFFN
from src.models.mbc_slices import MBCSemanticHead
from src.models.pcrg_token import PCRGTokenLayer
from src.models.position_bias import PositionBiasTower
from src.models.semantic_features import SemanticLongShortInterest, SimTierEncoder, VideoSemanticEncoder
from src.models.transformer_fusion import TransformerFusion


# ADS-Transformer-SideInfo：在 ADS 主干上增加行为侧信息、Transformer、Side Attention Bias、Dense/MBC 等功能。
class ADSTransformerSideModel(ADSModel):
    def __init__(self, config: Dict, feature_maps: Dict):
        super().__init__(config, feature_maps)
        # 主功能开关：控制序列 Transformer、行为侧信息和 Side Attention Bias 路径。
        self.use_transformer = bool(config.get("use_transformer", True))
        self.use_behavior_side = bool(config.get("use_behavior_side", True))

        side_dim = int(config.get("side_dim", 16))
        side_output_dim = int(config.get("side_output_dim", 32))
        d_model = int(config.get("transformer_d_model", 64))

        bucket_sizes = feature_maps.get("bucket_sizes", {})
        vocab_sizes = feature_maps["vocab_sizes"]

        # 行为侧信息由动作向量、播放比例、时间间隔、历史 tab 和位置编码组成。
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

        self.side_info_proj = nn.Linear(side_dim * 5, side_output_dim)
        self.h_input_proj = nn.Linear(self.d_s, d_model)
        self.side_to_model = nn.Linear(side_output_dim, d_model)
        self.side_gate = nn.Linear(d_model, d_model)
        self.side_post_norm = nn.LayerNorm(d_model)
        # Side Attention Bias 让侧信息直接作为注意力分数偏置影响历史行为权重。
        self.use_ads_kv_backbone = bool(config.get("use_ads_kv_backbone", True))
        self.use_side_attention_bias = bool(config.get("use_side_attention_bias", False))
        self.use_side_attention_interactions = bool(config.get("use_side_attention_interactions", False))
        self.use_side_attention_match_features = bool(config.get("use_side_attention_match_features", False))
        self.use_side_attention_gate = bool(config.get("use_side_attention_gate", False))
        self.side_attention_bias_scale = float(config.get("side_attention_bias_scale", 0.1))
        # attention_type=din_mlp 时使用 DIN 风格目标注意力，否则沿用点积注意力。
        self.attention_type = str(config.get("attention_type", "dot")).lower()
        self.din_attn = None
        if self.attention_type == "din_mlp":
            self.din_attn = DINStyleTargetAttention(
                d_attn=self.d_attn,
                attn_hidden_dim=int(config.get("attn_hidden_dim", 128)),
                dropout=float(config.get("dropout", 0.1)),
                use_side_bias=bool(config.get("use_side_bias", True)),
            )
        # PCRG Token 是多兴趣查询实验，默认关闭。
        self.use_pcrg_token = bool(config.get("use_pcrg_token", False))
        self.num_interest_tokens = int(config.get("num_interest_tokens", 4))
        self.pcrg_token = None
        self.pcrg_token_q_proj = None
        self.pcrg_token_score_mlp = None
        if self.use_pcrg_token:
            pcrg_token_dim = int(config.get("pcrg_token_dim", self.d_attn))
            self.pcrg_token = PCRGTokenLayer(
                target_dim=self.d_q,
                domain_dim=self.d_D,
                num_interest_tokens=self.num_interest_tokens,
                token_dim=pcrg_token_dim,
                hidden_dim=int(config.get("pcrg_token_hidden_dim", config.get("pcrg_hidden_dim", 128))),
            )
            self.pcrg_token_q_proj = nn.Linear(pcrg_token_dim, self.d_attn)
            self.pcrg_token_score_mlp = nn.Sequential(
                nn.Linear(self.d_attn * 4, int(config.get("attn_hidden_dim", 128))),
                nn.GELU(),
                nn.Dropout(float(config.get("dropout", 0.1))),
                nn.Linear(int(config.get("attn_hidden_dim", 128)), 1),
            )
        # TransformerFusion 对兴趣 token 做二次融合，默认关闭。
        self.use_transformer_fusion = bool(config.get("use_transformer_fusion", False))
        self.transformer_fusion = None
        self.fusion_gate_logit = None
        if self.use_transformer_fusion:
            self.transformer_fusion = TransformerFusion(
                dim=self.d_attn,
                num_heads=int(config.get("fusion_num_heads", 2)),
                num_layers=int(config.get("fusion_layers", 1)),
                ffn_dim=int(config.get("fusion_ffn_dim", 128)),
                dropout=float(config.get("dropout", 0.1)),
            )
            fusion_gate_init = float(config.get("fusion_gate_init", 0.1))
            fusion_gate_init = min(max(fusion_gate_init, 1e-4), 1.0 - 1e-4)
            self.fusion_gate_logit = nn.Parameter(
                torch.tensor(math.log(fusion_gate_init / (1.0 - fusion_gate_init)), dtype=torch.float32)
            )
        # 可学习标量门控用于控制 Side Attention Bias 强度。
        self.side_attention_gate_logit = None
        if self.use_side_attention_gate:
            side_attention_gate_init = float(config.get("side_attention_gate_init", 0.5))
            side_attention_gate_init = min(max(side_attention_gate_init, 1e-4), 1.0 - 1e-4)
            self.side_attention_gate_logit = nn.Parameter(
                torch.tensor(
                    math.log(side_attention_gate_init / (1.0 - side_attention_gate_init)),
                    dtype=torch.float32,
                )
            )
        # Side Attention Bias MLP 输入可拼接侧信息、Q/K、交互项、显式匹配特征和时间上下文。
        side_bias_hidden_dim = int(config.get("side_attention_bias_hidden_dim", 64))
        side_bias_input_dim = side_output_dim + self.d_attn * 2
        if self.use_side_attention_interactions:
            side_bias_input_dim += self.d_attn * 2
        if self.use_side_attention_match_features:
            side_bias_input_dim += 6
        if self.use_time_context:
            side_bias_input_dim += self.embedding_dim * 3
        self.side_attention_bias_mlp = nn.Sequential(
            nn.Linear(side_bias_input_dim, side_bias_hidden_dim),
            nn.GELU(),
            nn.Linear(side_bias_hidden_dim, 1),
        )
        if bool(config.get("zero_init_side_attention_bias", False)):
            nn.init.zeros_(self.side_attention_bias_mlp[-1].weight)
            nn.init.zeros_(self.side_attention_bias_mlp[-1].bias)

        kv_residual_init = float(config.get("transformer_kv_residual_init", 0.1))
        kv_residual_init = min(max(kv_residual_init, 1e-4), 1.0 - 1e-4)
        self.kv_residual_logit = nn.Parameter(
            torch.tensor(math.log(kv_residual_init / (1.0 - kv_residual_init)), dtype=torch.float32)
        )

        # 非 Side Attention Bias 路径下，Transformer 对历史序列进行上下文编码。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config.get("transformer_heads", 4)),
            dim_feedforward=int(config.get("transformer_ffn_dim", 256)),
            dropout=float(config.get("dropout", 0.1)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.get("transformer_layers", 1)),
        )

        self.linear_k_base = nn.Linear(self.d_s, self.d_attn)
        self.linear_v_base = nn.Linear(self.d_s, self.d_attn)
        self.linear_k_ctx = nn.Linear(d_model, self.d_attn)
        self.linear_v_ctx = nn.Linear(d_model, self.d_attn)

        # 轻量残差 FFN 用于增强聚合后的兴趣向量。
        self.interest_ffn = ResidualFFN(
            dim=self.d_attn,
            hidden_dim=int(config.get("interest_ffn_dim", 256)),
            dropout=float(config.get("dropout", 0.1)),
        )
        # Dense 特征分支：承接 video_stat/caption/dense history 等连续特征，并提供残差 logit。
        self.use_dense_features = (
            bool(config.get("use_dense_features", False))
            or bool(config.get("use_video_stat", False))
            or bool(config.get("use_caption", False))
        )
        self.dense_dim = int(feature_maps.get("dense_dim", 0))
        dense_hidden_dim = int(config.get("dense_hidden_dim", 64))
        self.dense_proj = None
        self.dense_residual_head = None
        if self.use_dense_features and self.dense_dim > 0:
            self.dense_proj = nn.Sequential(
                nn.LayerNorm(self.dense_dim),
                nn.Linear(self.dense_dim, dense_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("dropout", 0.1))),
            )
            self.dense_residual_head = nn.Linear(dense_hidden_dim, 1)
            final_in = self.d_attn + self.d_q + self.embedding_dim + self.d_D + dense_hidden_dim
            self.mlp = nn.Sequential(
                nn.Linear(final_in, 256),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, 1),
            )
        # MBC semantic slices：融合不同语义切片形成小残差头，是当前保护族的增强候选。
        self.use_mbc_slices = bool(config.get("use_mbc_slices", False))
        self.use_random_head = bool(config.get("use_random_head", False))
        self.use_position_bias_tower = bool(config.get("use_position_bias_tower", False))
        self.use_rank_calib_split = bool(config.get("use_rank_calib_split", False))
        self.use_history_dense_features = bool(config.get("use_history_dense_features", False))
        self.use_author_features = bool(config.get("use_author_features", False))
        self.use_author_prior = bool(config.get("use_author_prior", False))
        self.use_author_mbc_slice = bool(config.get("use_author_mbc_slice", False))
        self.use_long_short_interest = bool(config.get("use_long_short_interest", False))
        self.use_user_conditioned_mbc_gate = bool(config.get("use_user_conditioned_mbc_gate", False))
        self.use_video_semantic_emb = bool(config.get("use_video_semantic_emb", False))
        self.use_simtier_features = bool(config.get("use_simtier_features", False))
        self.use_semantic_long_short = bool(config.get("use_semantic_long_short", False))
        self.use_semantic_late_fusion = bool(config.get("use_semantic_late_fusion", False))
        self.semantic_inject = str(config.get("semantic_inject", "mbc_slice"))
        self.use_semantic_match_features = bool(config.get("use_semantic_match_features", False))
        self.mbc_semantic_head = None
        self.mbc_gate_logit = None
        self.semantic_input_dim = int(feature_maps.get("semantic_dim", 0)) or int(config.get("semantic_proj_input_dim", config.get("semantic_proj_dim", 64)))
        self.semantic_proj_dim = int(config.get("semantic_proj_dim", 64))
        self.simtier_input_dim = int(feature_maps.get("simtier_dim", 0))
        self.simtier_dim = int(config.get("simtier_dim", 64))
        self.semantic_target_scale = float(config.get("semantic_target_scale", 1.0))
        self.simtier_scale = float(config.get("simtier_scale", 1.0))
        self.use_semantic_slice_gates = bool(config.get("use_semantic_slice_gates", False))
        self.use_semantic_gate_regularization = bool(config.get("use_semantic_gate_regularization", False))
        self.semantic_target_gate_reg_target = float(config.get("semantic_target_gate_reg_target", config.get("semantic_target_gate_init", 1.0)))
        self.semantic_target_gate_reg_target = min(max(self.semantic_target_gate_reg_target, 1e-4), 1.0 - 1e-4)
        self.simtier_gate_reg_target = float(config.get("simtier_gate_reg_target", config.get("simtier_gate_init", 0.5)))
        self.simtier_gate_reg_target = min(max(self.simtier_gate_reg_target, 1e-4), 1.0 - 1e-4)
        self.semantic_target_gate_logit = None
        self.simtier_gate_logit = None
        self.semantic_encoder = None
        self.simtier_encoder = None
        self.semantic_long_short = None
        self.semantic_late_fusion_head = None
        self.semantic_late_fusion_gate_logit = None
        if self.use_video_semantic_emb:
            self.semantic_encoder = VideoSemanticEncoder(
                raw_dim=self.semantic_input_dim,
                semantic_proj_dim=self.semantic_proj_dim,
                semantic_dropout=float(config.get("semantic_dropout", config.get("dropout", 0.1))),
            )
        if self.use_simtier_features and self.simtier_input_dim > 0:
            self.simtier_encoder = SimTierEncoder(
                simtier_input_dim=self.simtier_input_dim,
                simtier_dim=self.simtier_dim,
                simtier_dropout=float(config.get("simtier_dropout", config.get("dropout", 0.1))),
            )
        if self.use_semantic_slice_gates:
            semantic_target_gate_init = float(config.get("semantic_target_gate_init", 1.0))
            semantic_target_gate_init = min(max(semantic_target_gate_init, 1e-4), 1.0 - 1e-4)
            simtier_gate_init = float(config.get("simtier_gate_init", 0.5))
            simtier_gate_init = min(max(simtier_gate_init, 1e-4), 1.0 - 1e-4)
            if self.semantic_encoder is not None:
                self.semantic_target_gate_logit = nn.Parameter(
                    torch.tensor(math.log(semantic_target_gate_init / (1.0 - semantic_target_gate_init)), dtype=torch.float32)
                )
            if self.simtier_encoder is not None:
                self.simtier_gate_logit = nn.Parameter(
                    torch.tensor(math.log(simtier_gate_init / (1.0 - simtier_gate_init)), dtype=torch.float32)
                )
        if self.use_semantic_long_short and self.use_video_semantic_emb:
            semantic_interest_dim = int(config.get("semantic_long_short_dim", self.semantic_proj_dim))
            self.semantic_long_short = SemanticLongShortInterest(
                raw_dim=self.semantic_input_dim,
                target_dim=self.semantic_proj_dim,
                interest_dim=semantic_interest_dim,
                gate_hidden_dim=int(config.get("semantic_long_short_gate_hidden_dim", 64)),
                short_history_len=int(config.get("short_history_len", 10)),
                history_order=str(config.get("history_order", "old_to_new")),
                simtier_dim=self.simtier_dim if self.simtier_encoder is not None else 0,
                target_repr_dim=self.d_q,
                domain_repr_dim=self.d_D,
                dropout=float(config.get("semantic_dropout", config.get("dropout", 0.1))),
            )
        if self.use_semantic_late_fusion:
            semantic_late_fusion_input_dim = self.d_attn + self.d_q + self.embedding_dim + self.d_D
            if self.semantic_encoder is not None:
                semantic_late_fusion_input_dim += self.semantic_proj_dim
            if self.simtier_encoder is not None:
                semantic_late_fusion_input_dim += self.simtier_dim
            if self.semantic_long_short is not None:
                semantic_late_fusion_input_dim += int(config.get("semantic_long_short_dim", self.semantic_proj_dim))
            semantic_late_fusion_hidden_dim = int(config.get("semantic_late_fusion_hidden_dim", 64))
            self.semantic_late_fusion_head = nn.Sequential(
                nn.LayerNorm(semantic_late_fusion_input_dim),
                nn.Linear(semantic_late_fusion_input_dim, semantic_late_fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("semantic_late_fusion_dropout", config.get("dropout", 0.1)))),
                nn.Linear(semantic_late_fusion_hidden_dim, 1),
            )
            semantic_late_fusion_gate_init = float(config.get("semantic_late_fusion_gate_init", 0.05))
            semantic_late_fusion_gate_init = min(max(semantic_late_fusion_gate_init, 1e-4), 1.0 - 1e-4)
            self.semantic_late_fusion_gate_logit = nn.Parameter(
                torch.tensor(
                    math.log(semantic_late_fusion_gate_init / (1.0 - semantic_late_fusion_gate_init)),
                    dtype=torch.float32,
                )
            )
        self.mbc_slice_dims = {
            "interest": self.d_attn,
            "target": self.d_q,
            "domain": self.d_D,
            "user": self.embedding_dim,
            "behavior_side": side_output_dim,
        }
        if self.semantic_encoder is not None and self.semantic_inject == "mbc_slice":
            self.mbc_slice_dims["semantic_target"] = self.semantic_proj_dim
        if self.simtier_encoder is not None and self.semantic_inject == "mbc_slice":
            self.mbc_slice_dims["simtier"] = self.simtier_dim
        if self.semantic_long_short is not None and self.semantic_inject == "mbc_slice":
            self.mbc_slice_dims["semantic_interest"] = int(config.get("semantic_long_short_dim", self.semantic_proj_dim))
        if self.use_mbc_slices:
            self.mbc_semantic_head = MBCSemanticHead(
                slice_dims=self.mbc_slice_dims,
                branch_dim=int(config.get("mbc_branch_dim", 128)),
                fusion_dim=int(config.get("mbc_fusion_dim", 64)),
                dropout=float(config.get("dropout", 0.1)),
            )
            mbc_gate_init = float(config.get("mbc_gate_init", 0.1))
            mbc_gate_init = min(max(mbc_gate_init, 1e-4), 1.0 - 1e-4)
            self.mbc_gate_logit = nn.Parameter(
                torch.tensor(math.log(mbc_gate_init / (1.0 - mbc_gate_init)), dtype=torch.float32)
            )
        self.mbc_dynamic_gate = None
        if self.use_mbc_slices and self.use_user_conditioned_mbc_gate:
            mbc_dynamic_hidden_dim = int(config.get("mbc_dynamic_gate_hidden_dim", 64))
            self.mbc_dynamic_gate = nn.Sequential(
                nn.Linear(sum(self.mbc_slice_dims.values()), mbc_dynamic_hidden_dim),
                nn.GELU(),
                nn.Linear(mbc_dynamic_hidden_dim, 1),
            )
        self.history_dense_dim = int(feature_maps.get("history_dense_dim", 0))
        self.history_dense_proj = None
        self.history_dense_emb_dim = int(config.get("history_dense_emb_dim", 64))
        if self.use_history_dense_features and self.history_dense_dim > 0:
            self.history_dense_proj = nn.Sequential(
                nn.LayerNorm(self.history_dense_dim),
                nn.Linear(self.history_dense_dim, self.history_dense_emb_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("history_dense_dropout", config.get("dropout", 0.1)))),
            )
        self.author_emb = None
        self.author_emb_dim = int(config.get("author_emb_dim", 16))
        self.use_author_history_match = bool(config.get("use_author_history_match", False))
        self.author_match_dim = 3 if self.use_author_features and self.use_author_history_match else 0
        if self.use_author_features and int(vocab_sizes.get("author_id", 0)) > 0:
            self.author_emb = nn.Embedding(int(vocab_sizes["author_id"]), self.author_emb_dim, padding_idx=0)
        self.author_mbc_proj = None
        if self.use_author_mbc_slice and self.author_emb is not None:
            self.author_mbc_proj = nn.Linear(self.author_emb_dim, self.d_q)
            nn.init.zeros_(self.author_mbc_proj.weight)
            nn.init.zeros_(self.author_mbc_proj.bias)
        self.author_prior_dim = int(feature_maps.get("author_prior_dim", 0))
        self.author_prior_emb_dim = int(config.get("author_prior_emb_dim", 16))
        self.author_prior_proj = None
        if self.use_author_prior and self.author_prior_dim > 0:
            self.author_prior_proj = nn.Sequential(
                nn.LayerNorm(self.author_prior_dim),
                nn.Linear(self.author_prior_dim, self.author_prior_emb_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("author_prior_dropout", config.get("dropout", 0.1)))),
            )
        self.long_short_gate = None
        if self.use_long_short_interest:
            self.long_short_gate = nn.Sequential(
                nn.Linear(self.d_q + self.embedding_dim + self.d_D, int(config.get("long_short_gate_hidden_dim", 64))),
                nn.GELU(),
                nn.Linear(int(config.get("long_short_gate_hidden_dim", 64)), 1),
            )
        self.position_bias_tower = None
        self.position_bias_scale = None
        if self.use_position_bias_tower:
            self.position_bias_tower = PositionBiasTower(
                feature_maps=feature_maps,
                embedding_dim=int(config.get("position_bias_embedding_dim", 8)),
                hidden_dims=[int(x) for x in config.get("position_bias_hidden_dims", [64, 32])],
                dropout=float(config.get("position_bias_dropout", config.get("dropout", 0.1))),
            )
            scale_init = float(config.get("position_bias_scale_init", 0.1))
            self.position_bias_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        self.calibration_head = None
        self.calib_scale = None
        if self.use_rank_calib_split:
            calib_dim = int(feature_maps.get("calibration_dense_dim", 0)) or self.dense_dim
            self.calibration_head = CalibrationHead(
                dense_dim=calib_dim,
                hidden_dims=[int(x) for x in config.get("calibration_hidden_dims", [64, 32])],
                dropout=float(config.get("calibration_dropout", config.get("dropout", 0.1))),
            )
            self.calib_scale = nn.Parameter(
                torch.tensor(float(config.get("calib_scale_init", 0.0)), dtype=torch.float32),
                requires_grad=bool(config.get("calib_scale_trainable", True)),
            )
        self.semantic_proj = None
        if self.use_video_semantic_emb and self.semantic_inject != "mbc_slice" and not self.use_semantic_late_fusion:
            self.semantic_proj = nn.Sequential(
                nn.LayerNorm(self.semantic_input_dim),
                nn.Linear(self.semantic_input_dim, self.semantic_proj_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("semantic_dropout", config.get("dropout", 0.1)))),
            )
        self.semantic_match_proj = None
        self.semantic_match_input_dim = int(feature_maps.get("semantic_match_dim", 0)) or int(config.get("semantic_match_feature_dim", 8))
        self.semantic_match_emb_dim = int(config.get("semantic_match_emb_dim", 32))
        if self.use_semantic_match_features:
            self.semantic_match_proj = nn.Sequential(
                nn.LayerNorm(self.semantic_match_input_dim),
                nn.Linear(self.semantic_match_input_dim, self.semantic_match_emb_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("semantic_match_dropout", config.get("dropout", 0.1)))),
            )
        final_in = self.d_attn + self.d_q + self.embedding_dim + self.d_D
        if self.use_dense_features and self.dense_proj is not None and not self.use_rank_calib_split:
            final_in += dense_hidden_dim
        if self.history_dense_proj is not None:
            final_in += self.history_dense_emb_dim
        if self.author_emb is not None:
            final_in += self.author_emb_dim
        if self.author_match_dim > 0:
            final_in += self.author_match_dim
        if self.author_prior_proj is not None:
            final_in += self.author_prior_emb_dim
        if self.semantic_proj is not None:
            final_in += self.semantic_proj_dim
        if not self.use_mbc_slices and self.semantic_encoder is not None and self.semantic_inject != "mbc_slice":
            final_in += self.semantic_proj_dim
        if not self.use_mbc_slices and self.simtier_encoder is not None and self.semantic_inject != "mbc_slice":
            final_in += self.simtier_dim
        if not self.use_mbc_slices and self.semantic_long_short is not None and self.semantic_inject != "mbc_slice":
            final_in += int(config.get("semantic_long_short_dim", self.semantic_proj_dim))
        if self.semantic_match_proj is not None:
            final_in += self.semantic_match_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(final_in, 256),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 1),
        )
        self.random_head = None
        if self.use_random_head:
            hidden_dims = [int(x) for x in config.get("random_head_hidden_dims", [128])]
            layers: list[nn.Module] = []
            in_dim = final_in
            for hidden_dim in hidden_dims:
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(float(config.get("random_head_dropout", 0.1)))])
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self.random_head = nn.Sequential(*layers)
        self._debug_shapes_logged = False

    def _append_dense_features(self, batch: Dict[str, torch.Tensor], final_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 连续特征一方面拼到最终 MLP 输入，另一方面通过 residual head 直接修正 logit。
        dense_residual = torch.zeros(final_input.size(0), device=final_input.device, dtype=final_input.dtype)
        if self.use_rank_calib_split:
            return final_input, dense_residual
        if self.use_dense_features and self.dense_proj is not None and "dense_features" in batch:
            dense_emb = self.dense_proj(batch["dense_features"].to(dtype=final_input.dtype))
            final_input = torch.cat([final_input, dense_emb], dim=-1)
            if self.dense_residual_head is not None:
                dense_residual = self.dense_residual_head(dense_emb).squeeze(-1)
        return final_input, dense_residual

    def _append_experimental_features(
        self,
        batch: Dict[str, torch.Tensor],
        final_input: torch.Tensor,
        semantic_outputs: Dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        history_dense_emb = None
        author_match_features = None
        author_prior_emb = None
        semantic_emb = None
        semantic_match_emb = None
        if self.history_dense_proj is not None and "history_dense_features" in batch:
            history_dense_emb = self.history_dense_proj(batch["history_dense_features"].to(dtype=final_input.dtype))
            final_input = torch.cat([final_input, history_dense_emb], dim=-1)
        if self.author_emb is not None and "target_author_id" in batch:
            author_id = batch["target_author_id"].long().clamp_min(0).clamp_max(self.author_emb.num_embeddings - 1)
            final_input = torch.cat([final_input, self.author_emb(author_id)], dim=-1)
        if self.author_match_dim > 0:
            author_match_features = self._build_author_match_features(batch, final_input.dtype)
            final_input = torch.cat([final_input, author_match_features], dim=-1)
        if self.author_prior_proj is not None and "author_prior_features" in batch:
            author_prior_emb = self.author_prior_proj(batch["author_prior_features"].to(dtype=final_input.dtype))
            final_input = torch.cat([final_input, author_prior_emb], dim=-1)
        if self.semantic_proj is not None:
            semantic_features = batch.get("target_semantic_emb")
            if semantic_features is None:
                semantic_features = final_input.new_zeros(final_input.size(0), self.semantic_input_dim)
            semantic_emb = self.semantic_proj(semantic_features.to(dtype=final_input.dtype))
            final_input = torch.cat([final_input, semantic_emb], dim=-1)
        if self.semantic_match_proj is not None:
            semantic_match_features = batch.get("semantic_match_features")
            if semantic_match_features is None:
                semantic_match_features = final_input.new_zeros(final_input.size(0), self.semantic_match_input_dim)
            semantic_match_emb = self.semantic_match_proj(semantic_match_features.to(dtype=final_input.dtype))
            final_input = torch.cat([final_input, semantic_match_emb], dim=-1)
        if not self.use_mbc_slices and self.semantic_inject != "mbc_slice" and semantic_outputs:
            append_parts = []
            if "semantic_target" in semantic_outputs:
                append_parts.append(semantic_outputs["semantic_target"])
            if "simtier" in semantic_outputs:
                append_parts.append(semantic_outputs["simtier"])
            if "semantic_interest" in semantic_outputs:
                append_parts.append(semantic_outputs["semantic_interest"])
            if append_parts:
                final_input = torch.cat([final_input] + append_parts, dim=-1)
        return final_input, history_dense_emb, author_match_features, author_prior_emb, semantic_emb, semantic_match_emb

    def _apply_output_heads(self, batch: Dict[str, torch.Tensor], final_input: torch.Tensor, ranking_logit: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {"logit": ranking_logit, "pred": torch.sigmoid(ranking_logit)}
        if self.use_random_head and self.random_head is not None:
            out["random_logit"] = self.random_head(final_input).squeeze(-1)
        if self.use_position_bias_tower and self.position_bias_tower is not None and self.position_bias_scale is not None:
            position_bias_logit = self.position_bias_tower(batch)
            observed_logit = ranking_logit + self.position_bias_scale * position_bias_logit
            out["relevance_logit"] = ranking_logit
            out["position_bias_logit"] = position_bias_logit
            out["observed_logit"] = observed_logit
            primary = str(self.config.get("pal_eval_primary_logit", "relevance")).lower()
            out["logit"] = observed_logit if primary == "observed" else ranking_logit
            out["pred"] = torch.sigmoid(out["logit"])
        if self.use_rank_calib_split and self.calibration_head is not None and self.calib_scale is not None:
            dense_features = batch.get("calibration_dense_features", batch.get("dense_features"))
            calibration_logit = self.calibration_head(dense_features, ranking_logit.size(0), ranking_logit.device, ranking_logit.dtype)
            final_logit = ranking_logit + self.calib_scale * calibration_logit
            out["ranking_logit"] = ranking_logit
            out["calibration_logit"] = calibration_logit
            out["final_logit"] = final_logit
            out["logit"] = final_logit
            out["pred"] = torch.sigmoid(final_logit)
        return out

    def _maybe_apply_long_short_interest(
        self,
        E_Q: torch.Tensor,
        E_D: torch.Tensor,
        user_emb: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        hist_mask: torch.Tensor,
        long_interest: torch.Tensor,
        score_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.use_long_short_interest or self.long_short_gate is None:
            return long_interest, None, None, None
        short_len = int(self.config.get("short_history_len", 10))
        positions = torch.arange(hist_mask.size(1), device=hist_mask.device).unsqueeze(0)
        short_mask = hist_mask * (positions >= max(hist_mask.size(1) - short_len, 0)).to(dtype=hist_mask.dtype)
        short_bias = score_bias * short_mask if score_bias is not None else None
        short_interest, _, _ = self._target_attention(Q, K, V, short_mask, score_bias=short_bias)
        gate = torch.sigmoid(self.long_short_gate(torch.cat([E_Q, user_emb, E_D], dim=-1)))
        interest = gate * short_interest + (1.0 - gate) * long_interest
        return interest, short_interest, long_interest, gate

    def _apply_mbc_slices(
        self,
        interest: torch.Tensor,
        E_Q: torch.Tensor,
        E_D: torch.Tensor,
        user_emb: torch.Tensor,
        side_info: torch.Tensor,
        hist_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        semantic_outputs: Dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        # MBC slices 将 interest/target/domain/user/behavior_side 汇总为辅助残差预测。
        residual = torch.zeros(interest.size(0), device=interest.device, dtype=interest.dtype)
        if not self.use_mbc_slices or self.mbc_semantic_head is None or self.mbc_gate_logit is None:
            return residual, None, None, None, None
        denom = hist_mask.sum(dim=1, keepdim=True).clamp_min(1.0).to(dtype=side_info.dtype)
        side_pool = (side_info * hist_mask.unsqueeze(-1).to(dtype=side_info.dtype)).sum(dim=1) / denom
        author_mbc_delta = None
        target_slice = E_Q
        if self.author_mbc_proj is not None and "target_author_id" in batch:
            author_id = batch["target_author_id"].long().clamp_min(0).clamp_max(self.author_emb.num_embeddings - 1)
            author_mbc_delta = self.author_mbc_proj(self.author_emb(author_id)).to(dtype=E_Q.dtype)
            target_slice = E_Q + author_mbc_delta
        feature_slices = {
            "interest": interest,
            "target": target_slice,
            "domain": E_D,
            "user": user_emb,
            "behavior_side": side_pool,
        }
        if semantic_outputs:
            if "semantic_target" in self.mbc_slice_dims and semantic_outputs.get("semantic_target") is not None:
                feature_slices["semantic_target"] = semantic_outputs["semantic_target"]
            if "simtier" in self.mbc_slice_dims and semantic_outputs.get("simtier") is not None:
                feature_slices["simtier"] = semantic_outputs["simtier"]
            if "semantic_interest" in self.mbc_slice_dims and semantic_outputs.get("semantic_interest") is not None:
                feature_slices["semantic_interest"] = semantic_outputs["semantic_interest"]
        z_mbc, mbc_logit = self.mbc_semantic_head(feature_slices)
        if self.mbc_dynamic_gate is not None:
            gate_input = torch.cat([feature_slices[name] for name in self.mbc_semantic_head.slice_names], dim=-1)
            temperature = max(float(self.config.get("mbc_gate_temperature", 1.0)), 1e-6)
            mbc_gate = torch.sigmoid(self.mbc_dynamic_gate(gate_input).squeeze(-1) / temperature)
        else:
            mbc_gate = torch.sigmoid(self.mbc_gate_logit)
        residual = mbc_gate * mbc_logit
        return residual, z_mbc, mbc_logit, mbc_gate, author_mbc_delta

    def _apply_semantic_late_fusion(
        self,
        interest: torch.Tensor,
        E_Q: torch.Tensor,
        E_D: torch.Tensor,
        user_emb: torch.Tensor,
        semantic_outputs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        residual = torch.zeros(interest.size(0), device=interest.device, dtype=interest.dtype)
        if (
            not self.use_semantic_late_fusion
            or self.semantic_late_fusion_head is None
            or self.semantic_late_fusion_gate_logit is None
        ):
            return residual, None, None
        parts = [interest, E_Q, user_emb, E_D]
        if self.semantic_encoder is not None:
            semantic_target = semantic_outputs.get("semantic_target")
            if semantic_target is None:
                semantic_target = interest.new_zeros(interest.size(0), self.semantic_proj_dim)
            parts.append(semantic_target)
        if self.simtier_encoder is not None:
            simtier = semantic_outputs.get("simtier")
            if simtier is None:
                simtier = interest.new_zeros(interest.size(0), self.simtier_dim)
            parts.append(simtier)
        if self.semantic_long_short is not None:
            semantic_interest = semantic_outputs.get("semantic_interest")
            if semantic_interest is None:
                semantic_interest = interest.new_zeros(
                    interest.size(0),
                    int(self.config.get("semantic_long_short_dim", self.semantic_proj_dim)),
                )
            parts.append(semantic_interest)
        semantic_late_fusion_input = torch.cat(parts, dim=-1)
        semantic_late_fusion_logit = self.semantic_late_fusion_head(semantic_late_fusion_input).squeeze(-1)
        semantic_late_fusion_gate = torch.sigmoid(self.semantic_late_fusion_gate_logit).to(dtype=semantic_late_fusion_logit.dtype)
        residual = semantic_late_fusion_gate * semantic_late_fusion_logit
        semantic_outputs["semantic_late_fusion_gate"] = semantic_late_fusion_gate.expand(interest.size(0))
        semantic_outputs["semantic_late_fusion_logit"] = semantic_late_fusion_logit
        return residual, semantic_late_fusion_logit, semantic_late_fusion_gate

    def _build_side_info(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 历史行为侧信息：描述每个历史位置的行为强度、时间间隔、来源 tab 和位置。
        hist_action = batch["hist_action_vector"]  # [B, T, 7]
        hist_play_ratio_bucket = batch["hist_play_ratio_bucket"]  # [B, T]
        hist_time_gap_bucket = batch["hist_time_gap_bucket"]  # [B, T]
        hist_tab = batch["hist_tab"]  # [B, T]
        hist_mask = batch["hist_mask"]  # [B, T]

        B, T = hist_tab.shape
        pos = torch.arange(1, T + 1, device=hist_tab.device).unsqueeze(0).expand(B, -1)  # [B, T]
        pos = pos * (hist_mask > 0).long()

        action_emb = self.action_linear(hist_action)  # [B, T, side_dim]
        play_ratio_emb = self.play_ratio_emb(hist_play_ratio_bucket)  # [B, T, side_dim]
        time_gap_emb = self.time_gap_emb(hist_time_gap_bucket)  # [B, T, side_dim]
        tab_emb = self.hist_tab_emb(hist_tab)  # [B, T, side_dim]
        pos_emb = self.position_emb(pos)  # [B, T, side_dim]

        side_info = torch.cat(
            [action_emb, play_ratio_emb, time_gap_emb, tab_emb, pos_emb], dim=-1
        )  # [B, T, 5*side_dim]

        side_info = self.side_info_proj(side_info)  # [B, T, side_output_dim]
        return side_info

    def _build_side_match_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 显式目标-历史匹配特征：类别、tab、时长是否一致，供 side bias 使用。
        hist_mask = batch["hist_mask"]
        features = [
            batch["hist_category_l1"].eq(batch["target_category_l1"].unsqueeze(1)),
            batch["hist_category_l2"].eq(batch["target_category_l2"].unsqueeze(1)),
            batch["hist_category_l3"].eq(batch["target_category_l3"].unsqueeze(1)),
            batch["hist_category_l4"].eq(batch["target_category_l4"].unsqueeze(1)),
            batch["hist_tab"].eq(batch["tab"].unsqueeze(1)),
            batch["hist_duration_bucket"].eq(batch["target_duration_bucket"].unsqueeze(1)),
        ]
        return torch.stack(features, dim=-1).to(dtype=hist_mask.dtype) * hist_mask.unsqueeze(-1)

    def _build_author_match_features(self, batch: Dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        hist_author = batch.get("hist_author_id")
        target_author = batch.get("target_author_id")
        hist_mask = batch["hist_mask"].to(dtype=dtype)
        if hist_author is None or target_author is None:
            return hist_mask.new_zeros(hist_mask.size(0), self.author_match_dim)
        same_author = hist_author.eq(target_author.unsqueeze(1)).to(dtype=dtype) * hist_mask
        denom = hist_mask.sum(dim=1).clamp_min(1.0)
        match_ratio = same_author.sum(dim=1) / denom
        recent_match = same_author[:, -1]
        any_match = (same_author.sum(dim=1) > 0).to(dtype=dtype)
        return torch.stack([match_ratio, recent_match, any_match], dim=-1)

    def _build_semantic_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        hist_mask: torch.Tensor,
        E_Q: torch.Tensor,
        E_D: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        target_semantic_repr = None
        simtier_repr = None
        if self.semantic_encoder is not None:
            target_semantic = batch.get("target_semantic_emb")
            if target_semantic is None:
                target_semantic = E_Q.new_zeros(E_Q.size(0), self.semantic_input_dim)
            target_semantic_repr = self.semantic_encoder(target_semantic.to(dtype=dtype))
            target_semantic_repr = target_semantic_repr * self.semantic_target_scale
            if self.semantic_target_gate_logit is not None:
                semantic_target_gate = torch.sigmoid(self.semantic_target_gate_logit).to(dtype=target_semantic_repr.dtype)
                target_semantic_repr = target_semantic_repr * semantic_target_gate
                outputs["semantic_target_slice_gate"] = semantic_target_gate.expand(E_Q.size(0))
            outputs["semantic_target"] = target_semantic_repr
        if self.simtier_encoder is not None:
            simtier_features = batch.get("simtier_features")
            if simtier_features is None:
                simtier_features = E_Q.new_zeros(E_Q.size(0), self.simtier_input_dim)
            simtier_repr = self.simtier_encoder(simtier_features.to(dtype=dtype))
            simtier_repr = simtier_repr * self.simtier_scale
            if self.simtier_gate_logit is not None:
                simtier_gate = torch.sigmoid(self.simtier_gate_logit).to(dtype=simtier_repr.dtype)
                simtier_repr = simtier_repr * simtier_gate
                outputs["simtier_slice_gate"] = simtier_gate.expand(E_Q.size(0))
            outputs["simtier"] = simtier_repr
        if self.semantic_long_short is not None and target_semantic_repr is not None:
            hist_semantic = batch.get("hist_semantic_emb")
            if hist_semantic is None:
                hist_semantic = E_Q.new_zeros(E_Q.size(0), hist_mask.size(1), self.semantic_input_dim)
            long_short_out = self.semantic_long_short(
                target_semantic_repr=target_semantic_repr,
                hist_semantic_emb=hist_semantic.to(dtype=dtype),
                hist_mask=hist_mask.to(dtype=dtype),
                simtier_repr=simtier_repr,
                target_repr=E_Q,
                domain_repr=E_D,
            )
            outputs.update(long_short_out)
        return outputs

    def _attach_semantic_gate_regularization(self, out: Dict[str, torch.Tensor]) -> None:
        if not self.use_semantic_gate_regularization:
            return
        losses = []
        if self.semantic_target_gate_logit is not None:
            gate = torch.sigmoid(self.semantic_target_gate_logit)
            target = gate.new_tensor(self.semantic_target_gate_reg_target)
            losses.append((gate - target).pow(2))
        if self.simtier_gate_logit is not None:
            gate = torch.sigmoid(self.simtier_gate_logit)
            target = gate.new_tensor(self.simtier_gate_reg_target)
            losses.append((gate - target).pow(2))
        if losses:
            out["semantic_gate_regularization_loss"] = torch.stack(losses).mean()

    def _attach_semantic_diagnostics(
        self,
        out: Dict[str, torch.Tensor],
        semantic_outputs: Dict[str, torch.Tensor],
    ) -> None:
        diagnostics: Dict[str, torch.Tensor] = {}
        if "semantic_gate" in semantic_outputs:
            diagnostics["semantic_gate"] = semantic_outputs["semantic_gate"].detach()
        if "short_history_non_empty" in semantic_outputs:
            diagnostics["short_history_non_empty"] = semantic_outputs["short_history_non_empty"].detach()
        if "long_history_non_empty" in semantic_outputs:
            diagnostics["long_history_non_empty"] = semantic_outputs["long_history_non_empty"].detach()
        if "short_sem_interest" in semantic_outputs:
            diagnostics["short_sem_interest_norm"] = semantic_outputs["short_sem_interest"].detach().norm(dim=-1)
        if "long_sem_interest" in semantic_outputs:
            diagnostics["long_sem_interest_norm"] = semantic_outputs["long_sem_interest"].detach().norm(dim=-1)
        if "semantic_target" in semantic_outputs:
            diagnostics["target_semantic_repr_norm"] = semantic_outputs["semantic_target"].detach().norm(dim=-1)
        if "semantic_target_slice_gate" in semantic_outputs:
            diagnostics["semantic_target_slice_gate"] = semantic_outputs["semantic_target_slice_gate"].detach()
        if "simtier_slice_gate" in semantic_outputs:
            diagnostics["simtier_slice_gate"] = semantic_outputs["simtier_slice_gate"].detach()
        if "semantic_late_fusion_gate" in semantic_outputs:
            diagnostics["semantic_late_fusion_gate"] = semantic_outputs["semantic_late_fusion_gate"].detach()
        if "semantic_late_fusion_logit" in semantic_outputs:
            diagnostics["semantic_late_fusion_logit"] = semantic_outputs["semantic_late_fusion_logit"].detach()
        if diagnostics:
            out["semantic_diagnostics"] = diagnostics

    def _build_side_time_features(self, batch: Dict[str, torch.Tensor], hist_mask: torch.Tensor) -> torch.Tensor:
        # 时间上下文会广播到每个历史位置，用于增强 side bias 的场景感知。
        time_feats = [
            self.hour_emb(batch["hour_of_day"]),
            self.dow_emb(batch["day_of_week"]),
            self.weekend_emb(batch["is_weekend"]),
        ]
        time_emb = torch.cat(time_feats, dim=-1).unsqueeze(1).expand(-1, hist_mask.size(1), -1)
        return time_emb * hist_mask.unsqueeze(-1)

    def _target_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        hist_mask: torch.Tensor,
        score_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # 统一封装目标注意力：根据配置选择 DIN MLP 注意力或点积注意力。
        if self.din_attn is not None:
            interest, alpha, score = self.din_attn(Q, K, V, hist_mask, score_bias=score_bias)
            return interest, alpha, score
        interest, alpha = self.attn(Q, K, V, hist_mask, score_bias=score_bias)
        return interest, alpha, None

    def _pcrg_token_attention(
        self,
        E_Q: torch.Tensor,
        E_D: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        hist_mask: torch.Tensor,
        score_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 多兴趣 token 注意力：每个 token 独立关注历史，再平均成最终兴趣。
        token_queries, token_mask = self.pcrg_token(E_Q, E_D)
        Q_token = self.pcrg_token_q_proj(token_queries)
        Q_exp = Q_token.unsqueeze(2).expand(-1, -1, K.size(1), -1)
        K_exp = K.unsqueeze(1).expand(-1, Q_token.size(1), -1, -1)
        V_exp = V.unsqueeze(1)
        attn_feat = torch.cat([Q_exp, K_exp, Q_exp - K_exp, Q_exp * K_exp], dim=-1)
        score = self.pcrg_token_score_mlp(attn_feat).squeeze(-1)
        if score_bias is not None:
            score = score + score_bias.unsqueeze(1)
        mask = hist_mask.unsqueeze(1)
        score = score.masked_fill(mask <= 0, -1e9)
        all_zero_mask = hist_mask.sum(dim=1) <= 0
        alpha = torch.softmax(score, dim=-1)
        alpha = alpha * (mask > 0).float()
        alpha_denom = alpha.sum(dim=-1, keepdim=True)
        alpha = torch.where(alpha_denom > 0, alpha / alpha_denom, torch.zeros_like(alpha))
        if all_zero_mask.any():
            alpha = alpha.clone()
            alpha[all_zero_mask] = 0.0
        token_interest = (alpha.unsqueeze(-1) * V_exp).sum(dim=2)
        token_interest = token_interest * token_mask.unsqueeze(-1).to(dtype=token_interest.dtype)
        token_denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0).to(dtype=token_interest.dtype)
        interest = token_interest.sum(dim=1) / token_denom
        return interest, alpha.mean(dim=1), score.mean(dim=1)

    def _apply_transformer_fusion(
        self,
        interest: torch.Tensor,
        Q: torch.Tensor,
        V: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 可选 TransformerFusion：用门控残差把二次融合兴趣加回主兴趣。
        if not self.use_transformer_fusion or self.transformer_fusion is None or self.fusion_gate_logit is None:
            return interest, None
        target_query = Q.mean(dim=1)
        fused_interest, fusion_alpha = self.transformer_fusion(target_query, V, hist_mask)
        gate = torch.sigmoid(self.fusion_gate_logit)
        return interest + gate * fused_interest, fusion_alpha

    def _maybe_log_debug_shapes(
        self,
        batch: Dict[str, torch.Tensor],
        E_S: torch.Tensor,
        E_D: torch.Tensor,
        E_Q: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        hist_mask: torch.Tensor,
        interest: torch.Tensor,
        final_input: torch.Tensor,
        logit: torch.Tensor,
        score_bias: torch.Tensor | None = None,
        attn_score: torch.Tensor | None = None,
        mbc_vector: torch.Tensor | None = None,
        mbc_logit: torch.Tensor | None = None,
        mbc_gate: torch.Tensor | None = None,
        random_logit: torch.Tensor | None = None,
        relevance_logit: torch.Tensor | None = None,
        position_bias_logit: torch.Tensor | None = None,
        observed_logit: torch.Tensor | None = None,
        ranking_logit: torch.Tensor | None = None,
        calibration_logit: torch.Tensor | None = None,
        final_logit: torch.Tensor | None = None,
        history_dense_emb: torch.Tensor | None = None,
        author_match_features: torch.Tensor | None = None,
        author_prior_emb: torch.Tensor | None = None,
        semantic_emb: torch.Tensor | None = None,
        semantic_match_emb: torch.Tensor | None = None,
        semantic_outputs: Dict[str, torch.Tensor] | None = None,
        short_interest: torch.Tensor | None = None,
        long_interest: torch.Tensor | None = None,
        long_short_gate: torch.Tensor | None = None,
        author_mbc_delta: torch.Tensor | None = None,
    ) -> None:
        if not bool(self.config.get("_debug_shapes", False)) or self._debug_shapes_logged:
            return
        parts = [
            f"hist_mask={tuple(hist_mask.shape)}",
            f"E_S={tuple(E_S.shape)}",
            f"E_D={tuple(E_D.shape)}",
            f"E_Q={tuple(E_Q.shape)}",
            f"Q={tuple(Q.shape)}",
            f"K={tuple(K.shape)}",
            f"V={tuple(V.shape)}",
            f"interest={tuple(interest.shape)}",
            f"final_input={tuple(final_input.shape)}",
            f"logit={tuple(logit.shape)}",
        ]
        if score_bias is not None:
            parts.append(f"side_bias={tuple(score_bias.shape)}")
        if attn_score is not None:
            parts.append(f"attention_score={tuple(attn_score.shape)}")
            if self.attention_type == "din_mlp":
                parts.append(f"din_attn_feat={(Q.size(0), Q.size(1), Q.size(2) * 4)}")
        if mbc_vector is not None:
            parts.append(f"mbc_vector={tuple(mbc_vector.shape)}")
        if mbc_logit is not None:
            parts.append(f"mbc_logit={tuple(mbc_logit.shape)}")
        if mbc_gate is not None:
            if mbc_gate.dim() == 0:
                parts.append(f"mbc_gate=scalar({float(mbc_gate.detach().cpu()):.6f})")
            else:
                parts.append(f"mbc_gate={tuple(mbc_gate.shape)}")
        if random_logit is not None:
            parts.append(f"random_logit={tuple(random_logit.shape)}")
        if relevance_logit is not None:
            parts.append(f"relevance_logit={tuple(relevance_logit.shape)}")
        if position_bias_logit is not None:
            parts.append(f"position_bias_logit={tuple(position_bias_logit.shape)}")
        if observed_logit is not None:
            parts.append(f"observed_logit={tuple(observed_logit.shape)}")
            parts.append(f"position_bias_scale={float(self.position_bias_scale.detach().cpu()) if self.position_bias_scale is not None else 0.0:.6f}")
        if ranking_logit is not None:
            parts.append(f"ranking_logit={tuple(ranking_logit.shape)}")
        if calibration_logit is not None:
            parts.append(f"calibration_logit={tuple(calibration_logit.shape)}")
        if final_logit is not None:
            parts.append(f"final_logit={tuple(final_logit.shape)}")
            parts.append(f"calib_scale={float(self.calib_scale.detach().cpu()) if self.calib_scale is not None else 0.0:.6f}")
        if history_dense_emb is not None:
            parts.append(f"history_dense_features={tuple(batch['history_dense_features'].shape)}")
            parts.append(f"history_dense_emb={tuple(history_dense_emb.shape)}")
        if author_match_features is not None:
            parts.append(f"author_match_features={tuple(author_match_features.shape)}")
        if author_mbc_delta is not None:
            parts.append(f"author_mbc_delta={tuple(author_mbc_delta.shape)}")
        if author_prior_emb is not None:
            parts.append(f"author_prior_features={tuple(batch['author_prior_features'].shape)}")
            parts.append(f"author_prior_emb={tuple(author_prior_emb.shape)}")
        if semantic_emb is not None:
            semantic_features = batch.get("target_semantic_emb")
            semantic_shape = tuple(semantic_features.shape) if semantic_features is not None else (semantic_emb.size(0), self.semantic_input_dim)
            parts.append(f"target_semantic_emb={semantic_shape}")
            parts.append(f"semantic_emb={tuple(semantic_emb.shape)}")
        if semantic_outputs:
            if "semantic_target" in semantic_outputs:
                parts.append(f"target_semantic_repr={tuple(semantic_outputs['semantic_target'].shape)}")
            if "simtier" in semantic_outputs:
                parts.append(f"simtier_repr={tuple(semantic_outputs['simtier'].shape)}")
            if "short_sem_interest" in semantic_outputs:
                parts.append(f"short_sem_interest={tuple(semantic_outputs['short_sem_interest'].shape)}")
            if "long_sem_interest" in semantic_outputs:
                parts.append(f"long_sem_interest={tuple(semantic_outputs['long_sem_interest'].shape)}")
            if "semantic_interest" in semantic_outputs:
                parts.append(f"semantic_interest={tuple(semantic_outputs['semantic_interest'].shape)}")
            if "semantic_gate" in semantic_outputs:
                gate = semantic_outputs["semantic_gate"].detach()
                parts.append(f"semantic_gate mean/std={float(gate.mean().cpu()):.6f}/{float(gate.std(unbiased=False).cpu()):.6f}")
            parts.append(f"MBC slice keys={list(self.mbc_slice_dims.keys())}")
        if semantic_match_emb is not None:
            semantic_match_features = batch.get("semantic_match_features")
            semantic_match_shape = tuple(semantic_match_features.shape) if semantic_match_features is not None else (semantic_match_emb.size(0), self.semantic_match_input_dim)
            parts.append(f"semantic_match_features={semantic_match_shape}")
            parts.append(f"semantic_match_emb={tuple(semantic_match_emb.shape)}")
        if short_interest is not None and long_interest is not None and long_short_gate is not None:
            parts.append(f"short_interest={tuple(short_interest.shape)}")
            parts.append(f"long_interest={tuple(long_interest.shape)}")
            parts.append(f"long_short_gate={tuple(long_short_gate.shape)}")
        if self.use_time_context:
            parts.extend(
                [
                    f"hour_of_day={tuple(batch['hour_of_day'].shape)}",
                    f"day_of_week={tuple(batch['day_of_week'].shape)}",
                    f"is_weekend={tuple(batch['is_weekend'].shape)}",
                ]
            )
        print("DEBUG_MODEL_SHAPES | " + " | ".join(parts))
        self._debug_shapes_logged = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hist_mask = batch["hist_mask"]  # [B, T]

        # 1) E_S：历史行为序列表示。
        E_S = self._build_seq_item_emb(batch)  # [B, T, d_s]
        # 2) E_D：场景/域表示。
        E_D, E_D_raw = self._build_domain_emb(batch)  # [B, d_D]
        E_D_generation = E_D_raw if self.ads_generation_mode == "paper" else E_D
        # 3) E_Q：目标物品表示。
        E_Q = self._build_target_item_emb(batch)  # [B, d_q]

        # 4) PSRG：场景个性化历史表示。
        if self.use_psrg:
            E_S_personalized = self.psrg(E_S, E_D_generation)
        else:
            E_S_personalized = E_S
        E_S_personalized = E_S_personalized * hist_mask.unsqueeze(-1)

        # 5) PCRG：为每个历史位置生成目标相关 query。
        if self.use_pcrg:
            Q_personalized = self.pcrg(E_Q, E_D_generation)  # [B, T, d_q]
        else:
            Q_personalized = E_Q.unsqueeze(1).expand(-1, self.max_seq_len, -1)
        Q = self.linear_q(Q_personalized)  # [B, T, d_attn]

        # Side Attention Bias 路径：当前保护族主线，侧信息直接修正注意力分数。
        if self.use_side_attention_bias:
            K = self.linear_k(E_S_personalized) * hist_mask.unsqueeze(-1)  # [B, T, d_attn]
            V = self.linear_v(E_S_personalized) * hist_mask.unsqueeze(-1)  # [B, T, d_attn]
            if self.use_behavior_side:
                side_info = self._build_side_info(batch)  # [B, T, side_output_dim]
            else:
                side_info = torch.zeros(
                    E_S_personalized.size(0),
                    E_S_personalized.size(1),
                    self.side_info_proj.out_features,
                    device=E_S_personalized.device,
                    dtype=E_S_personalized.dtype,
                )
            # side bias 的基础输入是行为侧信息、query 和 key，可按实验开关追加更多特征。
            score_bias_parts = [side_info, Q, K]
            if self.use_side_attention_interactions:
                score_bias_parts.extend([Q * K, torch.abs(Q - K)])
            if self.use_side_attention_match_features:
                score_bias_parts.append(self._build_side_match_features(batch).to(dtype=Q.dtype))
            if self.use_time_context:
                score_bias_parts.append(self._build_side_time_features(batch, hist_mask).to(dtype=Q.dtype))
            score_bias_input = torch.cat(score_bias_parts, dim=-1)
            score_bias = self.side_attention_bias_mlp(score_bias_input).squeeze(-1)
            # tanh 和 scale 限制偏置幅度，避免侧信息直接压过主注意力分数。
            score_bias = self.side_attention_bias_scale * torch.tanh(score_bias)
            if self.use_side_attention_gate and self.side_attention_gate_logit is not None:
                score_bias = torch.sigmoid(self.side_attention_gate_logit) * score_bias
            score_bias = score_bias * hist_mask
            if self.use_pcrg_token:
                interest, alpha, attn_score = self._pcrg_token_attention(E_Q, E_D, K, V, hist_mask, score_bias=score_bias)
            else:
                interest, alpha, attn_score = self._target_attention(Q, K, V, hist_mask, score_bias=score_bias)
            interest, fusion_alpha = self._apply_transformer_fusion(interest, Q, V, hist_mask)
            if fusion_alpha is not None:
                alpha = fusion_alpha
            user_emb = self.user_id_emb(batch["user_id"])  # [B, emb]
            interest, short_interest, long_interest, long_short_gate = self._maybe_apply_long_short_interest(
                E_Q, E_D, user_emb, Q, K, V, hist_mask, interest, score_bias
            )
            interest = self.interest_ffn(interest)  # [B, d_attn]
            semantic_outputs = self._build_semantic_outputs(batch, hist_mask, E_Q, E_D, interest.dtype)
            mbc_residual, mbc_vector, mbc_logit, mbc_gate, author_mbc_delta = self._apply_mbc_slices(interest, E_Q, E_D, user_emb, side_info, hist_mask, batch, semantic_outputs)
            # 最终预测融合兴趣、目标、用户、场景，并可叠加 dense/MBC 残差。
            final_input = torch.cat([interest, E_Q, user_emb, E_D], dim=-1)
            final_input, dense_residual = self._append_dense_features(batch, final_input)
            final_input, history_dense_emb, author_match_features, author_prior_emb, semantic_emb, semantic_match_emb = self._append_experimental_features(batch, final_input, semantic_outputs)
            semantic_late_residual, _, _ = self._apply_semantic_late_fusion(
                interest, E_Q, E_D, user_emb, semantic_outputs
            )
            logit = self.mlp(final_input).squeeze(-1) + dense_residual + mbc_residual + semantic_late_residual  # [B]
            out = self._apply_output_heads(batch, final_input, logit)
            out["attn"] = alpha
            self._attach_semantic_diagnostics(out, semantic_outputs)
            self._attach_semantic_gate_regularization(out)
            self._maybe_log_debug_shapes(
                batch, E_S, E_D, E_Q, Q, K, V, hist_mask, interest, final_input, out["logit"],
                score_bias, attn_score, mbc_vector, mbc_logit, mbc_gate,
                random_logit=out.get("random_logit"),
                relevance_logit=out.get("relevance_logit"),
                position_bias_logit=out.get("position_bias_logit"),
                observed_logit=out.get("observed_logit"),
                ranking_logit=out.get("ranking_logit"),
                calibration_logit=out.get("calibration_logit"),
                final_logit=out.get("final_logit"),
                history_dense_emb=history_dense_emb,
                author_match_features=author_match_features,
                author_prior_emb=author_prior_emb,
                semantic_emb=semantic_emb,
                semantic_match_emb=semantic_match_emb,
                semantic_outputs=semantic_outputs,
                short_interest=short_interest,
                long_interest=long_interest,
                long_short_gate=long_short_gate,
                author_mbc_delta=author_mbc_delta,
            )
            if mbc_vector is not None and mbc_logit is not None:
                out["branch_vectors"] = {"mbc_semantic": mbc_vector}
                out["branch_logits"] = {"mbc_semantic": mbc_logit}
            return out

        # 5/6) Transformer for sequential context modeling (without side info)
        H_proj = self.h_input_proj(E_S_personalized)  # [B, T, d_model]

        if self.use_transformer:
            src_key_padding_mask = hist_mask <= 0  # [B, T]
            all_pad = src_key_padding_mask.all(dim=1)  # [B]
            if all_pad.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_pad, 0] = False
                H_proj = H_proj.clone()
                H_proj[all_pad, 0, :] = 0.0

            H_ctx = self.transformer(H_proj, src_key_padding_mask=src_key_padding_mask)  # [B, T, d_model]
            if all_pad.any():
                H_ctx = H_ctx.clone()
                H_ctx[all_pad] = 0.0
        else:
            H_ctx = H_proj

        H_ctx = H_ctx * hist_mask.unsqueeze(-1)  # [B, T, d_model]

        # 7) SideInfo post-Transformer residual gated injection
        if self.use_behavior_side:
            side_info = self._build_side_info(batch)  # [B, T, side_output_dim]
        else:
            side_info = torch.zeros(
                E_S_personalized.size(0),
                E_S_personalized.size(1),
                self.side_info_proj.out_features,
                device=E_S_personalized.device,
                dtype=E_S_personalized.dtype,
            )
        side_emb = self.side_to_model(side_info)  # [B, T, d_model]
        gate = torch.sigmoid(self.side_gate(side_emb)) * hist_mask.unsqueeze(-1)  # [B, T, d_model]
        H_with_side = self.side_post_norm(H_ctx + gate * side_emb) * hist_mask.unsqueeze(-1)  # [B, T, d_model]

        # 8) K/V: ADS backbone + Transformer residual enhancement
        K_ctx = self.linear_k_ctx(H_with_side)  # [B, T, d_attn]
        V_ctx = self.linear_v_ctx(H_with_side)  # [B, T, d_attn]

        if self.use_ads_kv_backbone:
            K_base = self.linear_k_base(E_S_personalized)  # [B, T, d_attn]
            V_base = self.linear_v_base(E_S_personalized)  # [B, T, d_attn]
            kv_alpha = torch.sigmoid(self.kv_residual_logit)
            K = K_base + kv_alpha * K_ctx
            V = V_base + kv_alpha * V_ctx
        else:
            K = K_ctx
            V = V_ctx

        K = K * hist_mask.unsqueeze(-1)
        V = V * hist_mask.unsqueeze(-1)

        # 13) attention
        if self.use_pcrg_token:
            interest, alpha, attn_score = self._pcrg_token_attention(E_Q, E_D, K, V, hist_mask)
        else:
            interest, alpha, attn_score = self._target_attention(Q, K, V, hist_mask)  # interest: [B, d_attn]
        interest, fusion_alpha = self._apply_transformer_fusion(interest, Q, V, hist_mask)
        if fusion_alpha is not None:
            alpha = fusion_alpha

        # 14) Interest FFN
        user_emb = self.user_id_emb(batch["user_id"])  # [B, emb]
        interest, short_interest, long_interest, long_short_gate = self._maybe_apply_long_short_interest(
            E_Q, E_D, user_emb, Q, K, V, hist_mask, interest, None
        )
        interest = self.interest_ffn(interest)  # [B, d_attn]

        # 15/16) final prediction
        semantic_outputs = self._build_semantic_outputs(batch, hist_mask, E_Q, E_D, interest.dtype)
        mbc_residual, mbc_vector, mbc_logit, mbc_gate, author_mbc_delta = self._apply_mbc_slices(
            interest, E_Q, E_D, user_emb, side_info, hist_mask, batch, semantic_outputs
        )
        final_input = torch.cat([interest, E_Q, user_emb, E_D], dim=-1)
        final_input, dense_residual = self._append_dense_features(batch, final_input)
        final_input, history_dense_emb, author_match_features, author_prior_emb, semantic_emb, semantic_match_emb = self._append_experimental_features(batch, final_input, semantic_outputs)
        semantic_late_residual, _, _ = self._apply_semantic_late_fusion(
            interest, E_Q, E_D, user_emb, semantic_outputs
        )
        logit = self.mlp(final_input).squeeze(-1) + dense_residual + mbc_residual + semantic_late_residual  # [B]
        out = self._apply_output_heads(batch, final_input, logit)
        out["attn"] = alpha
        self._attach_semantic_diagnostics(out, semantic_outputs)
        self._attach_semantic_gate_regularization(out)
        self._maybe_log_debug_shapes(
            batch, E_S, E_D, E_Q, Q, K, V, hist_mask, interest, final_input, out["logit"],
            None, attn_score, mbc_vector, mbc_logit, mbc_gate,
            random_logit=out.get("random_logit"),
            relevance_logit=out.get("relevance_logit"),
            position_bias_logit=out.get("position_bias_logit"),
            observed_logit=out.get("observed_logit"),
            ranking_logit=out.get("ranking_logit"),
            calibration_logit=out.get("calibration_logit"),
            final_logit=out.get("final_logit"),
            history_dense_emb=history_dense_emb,
            author_match_features=author_match_features,
            author_prior_emb=author_prior_emb,
            semantic_emb=semantic_emb,
            semantic_match_emb=semantic_match_emb,
            semantic_outputs=semantic_outputs,
            short_interest=short_interest,
            long_interest=long_interest,
            long_short_gate=long_short_gate,
            author_mbc_delta=author_mbc_delta,
        )
        if mbc_vector is not None and mbc_logit is not None:
            out["branch_vectors"] = {"mbc_semantic": mbc_vector}
            out["branch_logits"] = {"mbc_semantic": mbc_logit}
        return out
