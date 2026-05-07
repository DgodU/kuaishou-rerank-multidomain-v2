from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io import load_dataframe


def _as_array(value, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        return np.stack([np.asarray(x, dtype=dtype) for x in arr], axis=0)
    return arr.astype(dtype, copy=False)


class KuaiRandDataset(Dataset):
    def __init__(self, parquet_path: str | Path, config: Dict | None = None, feature_maps: Dict | None = None):
        self.parquet_path = Path(parquet_path)
        self.config = config or {}
        self.feature_maps = feature_maps or {}
        df = load_dataframe(self.parquet_path)

        self.scalar_long_cols = [
            "user_id",
            "target_video_id",
            "tab",
            "user_active_degree",
            "register_days_bucket",
            "fans_user_num_bucket",
            "follow_user_num_bucket",
            "friend_user_num_bucket",
            "target_category_l1",
            "target_category_l2",
            "target_category_l3",
            "target_category_l4",
            "target_video_type",
            "target_duration_bucket",
            "target_tag",
        ]
        self.scalar_long_cols.extend(["is_random", "hour_of_day", "day_of_week", "is_weekend", "display_position", "position_bucket", "target_author_id"])
        for c in self.scalar_long_cols:
            if c not in df.columns:
                df[c] = 0
        for i in range(18):
            col = f"onehot_feat{i}"
            if col in df.columns:
                self.scalar_long_cols.append(col)
        self.seq_long_cols = [
            "hist_video_id",
            "hist_category_l1",
            "hist_category_l2",
            "hist_category_l3",
            "hist_category_l4",
            "hist_duration_bucket",
            "hist_tab",
            "hist_play_ratio_bucket",
            "hist_time_gap_bucket",
        ]
        if "hist_author_id" in df.columns:
            self.seq_long_cols.append("hist_author_id")
        self.seq_float_cols = ["hist_mask"]
        self.action_col = "hist_action_vector"
        self.dense_cols = [c for c in df.columns if c.startswith("dense_")]
        self.history_dense_cols = [c for c in df.columns if c.startswith("history_dense_")]
        self.author_prior_cols = [c for c in df.columns if c.startswith("author_prior_")]
        self.calibration_dense_cols = [c for c in df.columns if c.startswith("calibration_dense_")]
        self.semantic_cols = [c for c in df.columns if c.startswith("target_semantic_emb_")]
        self.semantic_match_cols = [c for c in df.columns if c.startswith("semantic_match_")]
        self.simtier_cols = [c for c in df.columns if c.startswith("simtier_") or c.startswith("sim_") or c.startswith("recent") or c.startswith("long_view_sim") or c.startswith("high_play_ratio_sim") or c.startswith("same_cat") or c.startswith("same_author")][: int(self.feature_maps.get("simtier_dim", 10**9))]
        self.return_hist_semantic_emb = bool(config.get("use_semantic_long_short", False))
        self.length = len(df)
        self.arrays: Dict[str, np.ndarray] = {}
        self.semantic_emb_matrix: np.ndarray | None = None

        for c in self.scalar_long_cols:
            self.arrays[c] = df[c].to_numpy(dtype=np.int64, copy=True)
            del df[c]
        self.arrays["label"] = df["label"].to_numpy(dtype=np.float32, copy=True)
        del df["label"]
        if self.dense_cols:
            self.arrays["dense_features"] = df[self.dense_cols].to_numpy(dtype=np.float32, copy=True)
            df.drop(columns=self.dense_cols, inplace=True)
        if self.history_dense_cols:
            self.arrays["history_dense_features"] = df[self.history_dense_cols].to_numpy(dtype=np.float32, copy=True)
            df.drop(columns=self.history_dense_cols, inplace=True)
        if self.author_prior_cols:
            self.arrays["author_prior_features"] = df[self.author_prior_cols].to_numpy(dtype=np.float32, copy=True)
            df.drop(columns=self.author_prior_cols, inplace=True)
        if self.calibration_dense_cols:
            self.arrays["calibration_dense_features"] = df[self.calibration_dense_cols].to_numpy(dtype=np.float32, copy=True)
            df.drop(columns=self.calibration_dense_cols, inplace=True)
        if self.semantic_cols:
            self.arrays["target_semantic_emb"] = df[self.semantic_cols].to_numpy(dtype=np.float32, copy=True)
            df.drop(columns=self.semantic_cols, inplace=True)
        if self.semantic_match_cols:
            self.arrays["semantic_match_features"] = df[self.semantic_match_cols].to_numpy(dtype=np.float32, copy=True)
            df.drop(columns=self.semantic_match_cols, inplace=True)
        if self.simtier_cols:
            simtier = df[self.simtier_cols].to_numpy(dtype=np.float32, copy=True)
            self.arrays["simtier_features"] = np.nan_to_num(simtier, nan=0.0, posinf=0.0, neginf=0.0)
            df.drop(columns=self.simtier_cols, inplace=True)
        if "target_semantic_idx" in df.columns:
            self.arrays["target_semantic_idx"] = df["target_semantic_idx"].to_numpy(dtype=np.int64, copy=True)
            del df["target_semantic_idx"]
        if "semantic_missing_flag" in df.columns:
            self.arrays["semantic_missing_flag"] = df["semantic_missing_flag"].to_numpy(dtype=np.float32, copy=True)
            del df["semantic_missing_flag"]
        if "hist_semantic_idx_seq" in df.columns:
            self.arrays["hist_semantic_idx"] = np.stack(
                [_as_array(x, np.int64) for x in df["hist_semantic_idx_seq"].to_numpy()],
                axis=0,
            )
            del df["hist_semantic_idx_seq"]

        for c in self.seq_long_cols:
            self.arrays[c] = np.stack(
                [_as_array(x, np.int64) for x in df[c].to_numpy()],
                axis=0,
            )
            del df[c]

        for c in self.seq_float_cols:
            self.arrays[c] = np.stack(
                [_as_array(x, np.float32) for x in df[c].to_numpy()],
                axis=0,
            )
            del df[c]

        self.arrays[self.action_col] = np.stack(
            [_as_array(x, np.float32) for x in df[self.action_col].to_numpy()],
            axis=0,
        )
        del df[self.action_col]
        del df

        use_semantic = bool(self.config.get("use_video_semantic_emb", False))
        if use_semantic:
            matrix_path = self.parquet_path.parent / "semantic_emb_matrix.npy"
            if matrix_path.exists():
                self.semantic_emb_matrix = np.load(matrix_path).astype(np.float32)
                if "target_semantic_idx" not in self.arrays:
                    self.arrays["target_semantic_idx"] = np.zeros(self.length, dtype=np.int64)
                if "hist_semantic_idx" not in self.arrays:
                    self.arrays["hist_semantic_idx"] = np.zeros((self.length, int(self.config.get("max_seq_len", 50))), dtype=np.int64)
                if "semantic_missing_flag" not in self.arrays:
                    self.arrays["semantic_missing_flag"] = (self.arrays["target_semantic_idx"] == 0).astype(np.float32)
            elif bool(self.config.get("_debug_shapes", False)):
                dim = int(self.feature_maps.get("semantic_dim", 0)) or int(self.config.get("semantic_proj_input_dim", self.config.get("semantic_proj_dim", 64)))
                self.semantic_emb_matrix = np.zeros((1, dim), dtype=np.float32)
                self.arrays["target_semantic_idx"] = np.zeros(self.length, dtype=np.int64)
                self.arrays["hist_semantic_idx"] = np.zeros((self.length, int(self.config.get("max_seq_len", 50))), dtype=np.int64)
                self.arrays["semantic_missing_flag"] = np.ones(self.length, dtype=np.float32)
                print("semantic embedding matrix not found, semantic tensors are zero-filled in debug.")
            else:
                raise FileNotFoundError(f"Semantic config is enabled but matrix is missing: {matrix_path}")
            print(f"semantic_emb_matrix shape: {self.semantic_emb_matrix.shape}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.arrays.items()}
        if self.semantic_emb_matrix is not None:
            target_idx = int(item.get("target_semantic_idx", 0))
            hist_idx = np.asarray(item.get("hist_semantic_idx", np.zeros(int(self.config.get("max_seq_len", 50)), dtype=np.int64)), dtype=np.int64)
            target_idx = min(max(target_idx, 0), self.semantic_emb_matrix.shape[0] - 1)
            item["target_semantic_emb"] = self.semantic_emb_matrix[target_idx]
            if self.return_hist_semantic_emb:
                hist_idx = np.clip(hist_idx, 0, self.semantic_emb_matrix.shape[0] - 1)
                item["hist_semantic_emb"] = self.semantic_emb_matrix[hist_idx]
        return item


def kuairand_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        values = [x[k] for x in batch]
        first = values[0]
        if isinstance(first, np.ndarray):
            out[k] = torch.as_tensor(np.stack(values, axis=0))
        else:
            out[k] = torch.as_tensor(np.asarray(values))
    return out
