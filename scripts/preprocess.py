from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, load_yaml, save_dataframe, save_json, save_pickle


def resolve_raw_file(data_raw_dir: Path, filename: str) -> Path:
    cands = [
        data_raw_dir / filename,
        data_raw_dir / "KuaiRand-Pure" / "data" / filename,
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {filename} under {data_raw_dir}")


def fit_sparse_mapping(series: pd.Series) -> Dict[Any, int]:
    values = series.fillna("<unk>").astype(str)
    uniq = values.value_counts().index.tolist()
    return {v: i + 1 for i, v in enumerate(uniq)}


def apply_sparse_mapping(series: pd.Series, mapping: Dict[Any, int]) -> pd.Series:
    values = series.fillna("<unk>").astype(str)
    return values.map(mapping).fillna(0).astype(np.int64)


def fit_log_bucket_edges(series: pd.Series, num_bins: int = 20) -> list[float]:
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float64).values
    arr = np.clip(arr, a_min=0.0, a_max=None)
    arr = np.log1p(arr)
    p99 = float(np.quantile(arr, 0.99)) if arr.size > 0 else 1.0
    arr = np.clip(arr, 0.0, p99)

    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(arr, qs)).astype(np.float64)
    if edges.size < 2:
        edges = np.array([0.0, max(p99, 1.0)], dtype=np.float64)
    return edges.tolist()


def apply_log_bucket(series: pd.Series, edges: list[float]) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float64).values
    arr = np.clip(arr, a_min=0.0, a_max=None)
    arr = np.log1p(arr)
    arr = np.clip(arr, edges[0], edges[-1])
    bins = np.asarray(edges[1:-1], dtype=np.float64)
    bucket = np.digitize(arr, bins=bins, right=False) + 1
    return pd.Series(bucket.astype(np.int64), index=series.index)


def bucketize_fixed(values: np.ndarray, boundaries: list[float]) -> np.ndarray:
    bins = np.asarray(boundaries, dtype=np.float64)
    return np.digitize(values, bins=bins, right=False).astype(np.int64) + 1


def load_categories_for_videos(categories_path: Path, video_ids: set[str]) -> pd.DataFrame:
    usecols = [
        "final_video_id",
        "first_level_category_id",
        "second_level_category_id",
        "third_level_category_id",
        "fourth_level_category_id",
    ]
    chunks = []
    try:
        reader = pd.read_csv(categories_path, usecols=usecols, chunksize=500_000)
        for chunk in reader:
            chunk["final_video_id"] = chunk["final_video_id"].astype(str)
            sub = chunk[chunk["final_video_id"].isin(video_ids)]
            if len(sub) > 0:
                chunks.append(sub)
    except pd.errors.ParserError:
        for chunk in pd.read_csv(categories_path, usecols=usecols, chunksize=500_000, engine="python", on_bad_lines="skip"):
            chunk["final_video_id"] = chunk["final_video_id"].astype(str)
            sub = chunk[chunk["final_video_id"].isin(video_ids)]
            if len(sub) > 0:
                chunks.append(sub)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates("final_video_id")
    return out


HISTORY_DENSE_COLS = [
    "history_dense_hist_len",
    "history_dense_hist_click_rate",
    "history_dense_hist_long_view_rate",
    "history_dense_hist_like_rate",
    "history_dense_hist_follow_rate",
    "history_dense_hist_comment_rate",
    "history_dense_hist_forward_rate",
    "history_dense_hist_hate_rate",
    "history_dense_recent_5_click_rate",
    "history_dense_recent_5_long_view_rate",
    "history_dense_recent_10_click_rate",
    "history_dense_recent_10_long_view_rate",
    "history_dense_recent_20_click_rate",
    "history_dense_recent_20_long_view_rate",
    "history_dense_same_target_category_l1_hist_ratio",
    "history_dense_same_target_category_l1_click_rate",
    "history_dense_same_target_category_l1_long_view_rate",
    "history_dense_same_target_category_l2_hist_ratio",
    "history_dense_same_target_category_l2_click_rate",
    "history_dense_same_target_category_l2_long_view_rate",
    "history_dense_same_target_category_l3_hist_ratio",
    "history_dense_same_target_category_l3_click_rate",
    "history_dense_same_target_tab_hist_ratio",
    "history_dense_same_target_tab_click_rate",
    "history_dense_same_target_tab_long_view_rate",
    "history_dense_same_target_duration_bucket_hist_ratio",
    "history_dense_same_target_duration_bucket_click_rate",
    "history_dense_same_target_duration_bucket_long_view_rate",
    "history_dense_same_target_author_hist_ratio",
    "history_dense_same_target_author_click_rate",
    "history_dense_same_target_author_long_view_rate",
    "history_dense_decay_click_sum_tau_1d",
    "history_dense_decay_click_sum_tau_3d",
    "history_dense_decay_long_view_sum_tau_1d",
    "history_dense_decay_long_view_sum_tau_3d",
    "history_dense_decay_same_category_l1_click_sum_tau_3d",
    "history_dense_decay_same_author_click_sum_tau_3d",
]

POSITION_FIELD_CANDIDATES = [
    "position",
    "pos",
    "rank",
    "item_rank",
    "request_rank",
    "display_position",
    "display_pos",
    "page_position",
    "position_id",
    "index",
    "seq_position",
]

AUTHOR_FIELD_CANDIDATES = [
    "author_id",
    "user_id_of_video",
    "video_author_id",
    "creator_id",
    "upload_user_id",
    "photo_author_id",
]


def add_video_stat_dense_features(df: pd.DataFrame, source_cols: list[str]) -> list[str]:
    dense_cols = []
    for col in source_cols:
        if col not in df.columns:
            continue
        dense_col = f"dense_video_stat_{col}"
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float64).values
        values = np.clip(values, a_min=0.0, a_max=None)
        df[dense_col] = np.log1p(values).astype(np.float32)
        dense_cols.append(dense_col)
    return dense_cols


def add_caption_dense_features(df: pd.DataFrame, source_cols: list[str]) -> list[str]:
    dense_cols = []
    for col in source_cols:
        if col not in df.columns:
            continue
        dense_col = f"dense_caption_{col}"
        df[dense_col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
        dense_cols.append(dense_col)
    return dense_cols


def build_history_dense_values(row: Dict[str, Any], seq: list[Dict[str, Any]], max_seq_len: int) -> list[float]:
    n = len(seq)
    if n == 0:
        return [0.0] * len(HISTORY_DENSE_COLS)

    def smooth_rate(pos: float, cnt: float, alpha: float = 0.5, beta: float = 1.0) -> float:
        return float((pos + alpha) / (cnt + beta)) if cnt > 0 else 0.0

    def action_sum(index: int, items: list[Dict[str, Any]]) -> float:
        return float(sum(float(x["action_vec"][index]) for x in items))

    def rate(index: int, items: list[Dict[str, Any]]) -> float:
        return smooth_rate(action_sum(index, items), float(len(items)))

    def exp_sum(index: int, tau_days: float, items: list[Dict[str, Any]], same_key: str | None = None, same_value: int | None = None) -> float:
        denom = max(tau_days * 24.0 * 60.0 * 60.0 * 1000.0, 1.0)
        total = 0.0
        now = int(row["time_ms"])
        for x in items:
            if same_key is not None and int(x.get(same_key, 0)) != int(same_value):
                continue
            weight = float(np.exp(-max(now - int(x["time_ms"]), 0) / denom))
            total += weight * float(x["action_vec"][index])
        return float(np.clip(total, 0.0, 50.0) / 50.0)

    target_l1 = int(row["target_category_l1"])
    target_l2 = int(row["target_category_l2"])
    target_l3 = int(row["target_category_l3"])
    target_tab = int(row["tab"])
    target_duration = int(row["target_duration_bucket"])
    target_author = int(row.get("target_author_id", 0))
    same_l1 = [x for x in seq if int(x["target_category_l1"]) == target_l1]
    same_l2 = [x for x in seq if int(x["target_category_l2"]) == target_l2]
    same_l3 = [x for x in seq if int(x["target_category_l3"]) == target_l3]
    same_tab = [x for x in seq if int(x["tab"]) == target_tab]
    same_duration = [x for x in seq if int(x["target_duration_bucket"]) == target_duration]
    same_author = [x for x in seq if target_author > 0 and int(x.get("target_author_id", 0)) == target_author]
    recent5 = seq[-5:]
    recent10 = seq[-10:]
    recent20 = seq[-20:]
    values = [
        min(float(n) / float(max_seq_len), 1.0),
        rate(0, seq),
        rate(6, seq),
        rate(1, seq),
        rate(2, seq),
        rate(3, seq),
        rate(4, seq),
        rate(5, seq),
        rate(0, recent5),
        rate(6, recent5),
        rate(0, recent10),
        rate(6, recent10),
        rate(0, recent20),
        rate(6, recent20),
        float(len(same_l1)) / float(n),
        rate(0, same_l1),
        rate(6, same_l1),
        float(len(same_l2)) / float(n),
        rate(0, same_l2),
        rate(6, same_l2),
        float(len(same_l3)) / float(n),
        rate(0, same_l3),
        float(len(same_tab)) / float(n),
        rate(0, same_tab),
        rate(6, same_tab),
        float(len(same_duration)) / float(n),
        rate(0, same_duration),
        rate(6, same_duration),
        float(len(same_author)) / float(n),
        rate(0, same_author),
        rate(6, same_author),
        exp_sum(0, 1.0, seq),
        exp_sum(0, 3.0, seq),
        exp_sum(6, 1.0, seq),
        exp_sum(6, 3.0, seq),
        exp_sum(0, 3.0, seq, "target_category_l1", target_l1),
        exp_sum(0, 3.0, seq, "target_author_id", target_author) if target_author > 0 else 0.0,
    ]
    return [float(np.clip(x, 0.0, 1.0)) for x in values]


def add_train_target_priors(df: pd.DataFrame, train_end_date: int = 20220421, alpha: float = 20.0) -> list[str]:
    prior_specs = [
        ("user_id", "dense_user_prior"),
        ("target_video_id", "dense_video_prior"),
        ("target_category_l1", "dense_l1_prior"),
        ("target_category_l2", "dense_l2_prior"),
        ("tab", "dense_tab_prior"),
        ("target_duration_bucket", "dense_duration_prior"),
        ("user_active_degree", "dense_active_prior"),
    ]
    train_mask = df["date"] <= train_end_date
    train_df = df.loc[train_mask]
    global_ctr = float(train_df["label"].mean()) if len(train_df) > 0 else 0.0
    own_label = df["label"].astype(np.float64).values
    train_mask_arr = train_mask.values
    dense_cols = []

    for key, prefix in prior_specs:
        stat = train_df.groupby(key)["label"].agg(["sum", "count"])
        sums = df[key].map(stat["sum"]).fillna(0.0).astype(np.float64).values
        counts = df[key].map(stat["count"]).fillna(0.0).astype(np.float64).values
        sums = sums - np.where(train_mask_arr, own_label, 0.0)
        counts = counts - np.where(train_mask_arr, 1.0, 0.0)
        counts = np.maximum(counts, 0.0)
        rate = (sums + alpha * global_ctr) / (counts + alpha)
        max_count = float(max(stat["count"].max(), 1)) if len(stat) > 0 else 1.0
        count_feature = np.log1p(counts) / np.log1p(max_count)
        rate_col = f"{prefix}_ctr"
        count_col = f"{prefix}_count"
        df[rate_col] = rate.astype(np.float32)
        df[count_col] = count_feature.astype(np.float32)
        dense_cols.extend([rate_col, count_col])

    return dense_cols


def add_author_priors(df: pd.DataFrame, train_end_date: int = 20220426, alpha: float = 20.0) -> list[str]:
    train_mask = (df["date"] <= train_end_date) & (df["is_random"] == 0)
    train_df = df.loc[train_mask]
    global_ctr = float(train_df["label"].mean()) if len(train_df) > 0 else 0.0
    own_label = df["label"].astype(np.float64).values
    train_mask_arr = train_mask.values
    stat = train_df.groupby("target_author_id")["label"].agg(["sum", "count"])
    sums = df["target_author_id"].map(stat["sum"]).fillna(0.0).astype(np.float64).values
    counts = df["target_author_id"].map(stat["count"]).fillna(0.0).astype(np.float64).values
    sums = sums - np.where(train_mask_arr, own_label, 0.0)
    counts = counts - np.where(train_mask_arr, 1.0, 0.0)
    counts = np.maximum(counts, 0.0)
    rate = (sums + alpha * global_ctr) / (counts + alpha)
    max_count = float(max(stat["count"].max(), 1)) if len(stat) > 0 else 1.0
    count_feature = np.log1p(counts) / np.log1p(max_count)
    df["author_prior_ctr"] = rate.astype(np.float32)
    df["author_prior_count"] = count_feature.astype(np.float32)
    return ["author_prior_ctr", "author_prior_count"]


def detect_position_field(df: pd.DataFrame) -> str | None:
    for col in POSITION_FIELD_CANDIDATES:
        if col in df.columns:
            return col
    return None


def add_position_features(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    source = detect_position_field(df)
    if source is None:
        df["display_position"] = 0
        df["position_bucket"] = 0
        return df, {
            "position_source_field": None,
            "message": "No explicit position/rank field found. PAL will use context-only bias tower.",
            "position_bucket_counts": {},
        }
    pos = pd.to_numeric(df[source], errors="coerce").fillna(0).astype(np.int64).clip(lower=0)
    df["display_position"] = pos
    bounds = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]
    df["position_bucket"] = bucketize_fixed(pos.to_numpy(dtype=np.float64), bounds)
    counts = df["position_bucket"].value_counts().sort_index().to_dict()
    return df, {
        "position_source_field": source,
        "position_bucket_counts": {str(k): int(v) for k, v in counts.items()},
    }


def split_frames(df_model: pd.DataFrame, split_protocol: str) -> Dict[str, pd.DataFrame]:
    if split_protocol == "random_aux_split":
        return {
            "train": df_model[(df_model["is_random"] == 0) & (df_model["date"] <= 20220426)].copy(),
            "valid": df_model[(df_model["is_random"] == 0) & (df_model["date"] >= 20220427) & (df_model["date"] <= 20220430)].copy(),
            "test": df_model[(df_model["is_random"] == 0) & (df_model["date"] >= 20220501)].copy(),
            "random_aux_train": df_model[(df_model["is_random"] == 1) & (df_model["date"] >= 20220422) & (df_model["date"] <= 20220426)].copy(),
            "random_valid": df_model[(df_model["is_random"] == 1) & (df_model["date"] >= 20220427) & (df_model["date"] <= 20220430)].copy(),
            "random_test": df_model[(df_model["is_random"] == 1) & (df_model["date"] >= 20220501)].copy(),
        }
    return {
        "train": df_model[(df_model["is_random"] == 0) & (df_model["date"] <= 20220421)].copy(),
        "valid": df_model[(df_model["is_random"] == 0) & (df_model["date"] >= 20220422) & (df_model["date"] <= 20220430)].copy(),
        "test": df_model[(df_model["is_random"] == 0) & (df_model["date"] >= 20220501)].copy(),
        "random_aux_train": df_model.iloc[0:0].copy(),
        "random_valid": df_model.iloc[0:0].copy(),
        "random_test": df_model.iloc[0:0].copy(),
    }


def summarize_split(df: pd.DataFrame) -> Dict[str, Any]:
    out = summarize_hist_lengths(df)
    out["positive_rate"] = float(df["label"].mean()) if len(df) > 0 else 0.0
    if "position_bucket" in df.columns and len(df) > 0:
        grouped = df.groupby("position_bucket")["label"].mean()
        out["click_rate_by_position_bucket"] = {str(k): float(v) for k, v in grouped.items()}
    return out


def should_keep_history(row: Dict[str, Any], history_mode: str) -> bool:
    if history_mode == "all":
        return True
    if history_mode == "click_only":
        return int(row["is_click"]) == 1
    if history_mode == "click_or_long_view":
        return int(row["is_click"]) == 1 or int(row["long_view"]) == 1
    raise ValueError(f"Unsupported history_mode: {history_mode}")


def add_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hourmin" in df.columns:
        hour = pd.to_numeric(df["hourmin"], errors="coerce").fillna(-1).astype(np.int64) // 100
        hour = hour.where((hour >= 0) & (hour <= 23), -1)
    else:
        ts = pd.to_datetime(pd.to_numeric(df["time_ms"], errors="coerce"), unit="ms", errors="coerce")
        hour = ts.dt.hour.fillna(-1).astype(np.int64)

    date_dt = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    dow = date_dt.dt.dayofweek.fillna(-1).astype(np.int64)

    df["hour_of_day"] = np.where(hour >= 0, hour + 1, 0).astype(np.int64)
    df["day_of_week"] = np.where(dow >= 0, dow + 1, 0).astype(np.int64)
    df["is_weekend"] = np.where(dow < 0, 0, np.where(dow >= 5, 2, 1)).astype(np.int64)
    return df


def summarize_hist_lengths(df: pd.DataFrame) -> Dict[str, float | int]:
    if len(df) == 0:
        return {
            "sample_count": 0,
            "mean_hist_len": 0.0,
            "median_hist_len": 0.0,
            "p90_hist_len": 0.0,
            "p99_hist_len": 0.0,
            "hist_len_0_ratio": 0.0,
            "hist_len_lt_5_ratio": 0.0,
            "hist_len_eq_max_ratio": 0.0,
        }
    hist_len = np.asarray([float(np.asarray(x, dtype=np.float32).sum()) for x in df["hist_mask"].to_numpy()])
    max_len = float(hist_len.max()) if hist_len.size > 0 else 0.0
    return {
        "sample_count": int(len(df)),
        "mean_hist_len": float(np.mean(hist_len)),
        "median_hist_len": float(np.median(hist_len)),
        "p90_hist_len": float(np.quantile(hist_len, 0.90)),
        "p99_hist_len": float(np.quantile(hist_len, 0.99)),
        "hist_len_0_ratio": float(np.mean(hist_len == 0)),
        "hist_len_lt_5_ratio": float(np.mean(hist_len < 5)),
        "hist_len_eq_max_ratio": float(np.mean(hist_len == max_len)),
    }


def build_histories(
    df: pd.DataFrame,
    max_seq_len: int,
    use_dense_features: bool = False,
    history_mode: str = "all",
) -> pd.DataFrame:
    hist_cols = {
        "hist_video_id": [],
        "hist_category_l1": [],
        "hist_category_l2": [],
        "hist_category_l3": [],
        "hist_category_l4": [],
        "hist_duration_bucket": [],
        "hist_tab": [],
        "hist_author_id": [],
        "hist_action_vector": [],
        "hist_play_ratio_bucket": [],
        "hist_time_gap_bucket": [],
        "hist_mask": [],
    }
    if use_dense_features:
        for c in HISTORY_DENSE_COLS:
            hist_cols[c] = []

    time_gap_bounds = [
        0,
        60_000,
        5 * 60_000,
        30 * 60_000,
        60 * 60_000,
        6 * 60 * 60_000,
        24 * 60 * 60_000,
        3 * 24 * 60 * 60_000,
        7 * 24 * 60 * 60_000,
        14 * 24 * 60 * 60_000,
        30 * 24 * 60 * 60_000,
    ]

    grouped = df.groupby("user_id_raw", sort=False)

    for _, g in tqdm(grouped, total=grouped.ngroups, desc="Building offline histories"):
        g = g.sort_values("time_ms", kind="mergesort")
        records = list(g.to_dict("records"))
        hist_queue: deque[Dict[str, Any]] = deque(maxlen=max_seq_len)

        i = 0
        while i < len(records):
            current_ts = records[i]["time_ms"]
            j = i
            while j < len(records) and records[j]["time_ms"] == current_ts:
                j += 1

            snapshot = list(hist_queue)

            for k in range(i, j):
                row = records[k]
                seq = snapshot[-max_seq_len:]
                pad_len = max_seq_len - len(seq)

                hist_video_id = [x["target_video_id"] for x in seq]
                hist_c1 = [x["target_category_l1"] for x in seq]
                hist_c2 = [x["target_category_l2"] for x in seq]
                hist_c3 = [x["target_category_l3"] for x in seq]
                hist_c4 = [x["target_category_l4"] for x in seq]
                hist_dur = [x["target_duration_bucket"] for x in seq]
                hist_tab = [x["tab"] for x in seq]
                hist_author = [x.get("target_author_id", 0) for x in seq]
                hist_actions = [x["action_vec"] for x in seq]
                hist_play_ratio = [x["play_ratio_bucket"] for x in seq]

                time_gap_ms = [max(int(row["time_ms"]) - int(x["time_ms"]), 0) for x in seq]
                hist_time_gap = (
                    bucketize_fixed(np.asarray(time_gap_ms, dtype=np.float64), time_gap_bounds).tolist()
                    if len(time_gap_ms) > 0
                    else []
                )

                hist_cols["hist_video_id"].append([0] * pad_len + hist_video_id)
                hist_cols["hist_category_l1"].append([0] * pad_len + hist_c1)
                hist_cols["hist_category_l2"].append([0] * pad_len + hist_c2)
                hist_cols["hist_category_l3"].append([0] * pad_len + hist_c3)
                hist_cols["hist_category_l4"].append([0] * pad_len + hist_c4)
                hist_cols["hist_duration_bucket"].append([0] * pad_len + hist_dur)
                hist_cols["hist_tab"].append([0] * pad_len + hist_tab)
                hist_cols["hist_author_id"].append([0] * pad_len + hist_author)
                hist_cols["hist_action_vector"].append(
                    [[0.0] * 7 for _ in range(pad_len)] + hist_actions
                )
                hist_cols["hist_play_ratio_bucket"].append([0] * pad_len + hist_play_ratio)
                hist_cols["hist_time_gap_bucket"].append([0] * pad_len + hist_time_gap)
                hist_cols["hist_mask"].append([0.0] * pad_len + [1.0] * len(seq))
                if use_dense_features:
                    dense_values = build_history_dense_values(row, seq, max_seq_len)
                    for c, v in zip(HISTORY_DENSE_COLS, dense_values):
                        hist_cols[c].append(v)

            for k in range(i, j):
                row = records[k]
                if should_keep_history(row, history_mode):
                    hist_queue.append(
                        {
                            "target_video_id": int(row["target_video_id"]),
                            "target_category_l1": int(row["target_category_l1"]),
                            "target_category_l2": int(row["target_category_l2"]),
                            "target_category_l3": int(row["target_category_l3"]),
                            "target_category_l4": int(row["target_category_l4"]),
                            "target_duration_bucket": int(row["target_duration_bucket"]),
                            "tab": int(row["tab"]),
                            "target_author_id": int(row.get("target_author_id", 0)),
                            "action_vec": [
                                float(row["is_click"]),
                                float(row["is_like"]),
                                float(row["is_follow"]),
                                float(row["is_comment"]),
                                float(row["is_forward"]),
                                float(row["is_hate"]),
                                float(row["long_view"]),
                            ],
                            "play_ratio_bucket": int(row["play_ratio_bucket"]),
                            "time_ms": int(row["time_ms"]),
                        }
                    )

            i = j

    for k, v in hist_cols.items():
        df[k] = v
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--history_mode", type=str, default=None, choices=["all", "click_only", "click_or_long_view"])
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.history_mode is not None:
        config["history_mode"] = args.history_mode

    max_seq_len = int(config.get("max_seq_len", 50))
    use_video_stat = bool(config.get("use_video_stat", False))
    use_caption = bool(config.get("use_caption", False))
    use_dense_features = bool(config.get("use_dense_features", False))
    use_dense_history_only = bool(config.get("use_dense_history_only", False))
    use_history_dense_features = bool(config.get("use_history_dense_features", False))
    use_author_features = bool(config.get("use_author_features", False))
    use_author_prior = bool(config.get("use_author_prior", False))
    use_video_semantic_emb = bool(config.get("use_video_semantic_emb", False))
    use_semantic_match_features = bool(config.get("use_semantic_match_features", False))
    split_protocol = str(config.get("split_protocol", "original"))
    if use_dense_history_only:
        use_dense_features = True
    use_time_context = bool(config.get("use_time_context", False))
    if split_protocol == "random_aux_split":
        use_time_context = True
    history_mode = str(config.get("history_mode", "all"))

    data_raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = ensure_dir(PROJECT_ROOT / config.get("processed_dir", config.get("data_dir", "data/processed")))

    log1_path = resolve_raw_file(data_raw_dir, "log_standard_4_08_to_4_21_pure.csv")
    log2_path = resolve_raw_file(data_raw_dir, "log_standard_4_22_to_5_08_pure.csv")
    user_path = resolve_raw_file(data_raw_dir, "user_features_pure.csv")
    video_basic_path = resolve_raw_file(data_raw_dir, "video_features_basic_pure.csv")

    log1 = pd.read_csv(log1_path)
    log2 = pd.read_csv(log2_path)
    log1["is_random"] = 0
    log2["is_random"] = 0
    logs = [log1, log2]
    if split_protocol == "random_aux_split":
        random_path = resolve_raw_file(data_raw_dir, "log_random_4_22_to_5_08_pure.csv")
        random_log = pd.read_csv(random_path)
        random_log["is_random"] = 1
        logs.append(random_log)
    all_logs = pd.concat(logs, ignore_index=True)

    if "is_rand" in all_logs.columns and split_protocol != "random_aux_split":
        all_logs = all_logs[all_logs["is_rand"].fillna(0).astype(int) == 0].copy()
    all_logs["is_random"] = all_logs["is_random"].fillna(0).astype(np.int8)

    all_logs["label"] = all_logs["is_click"].astype(np.float32)
    all_logs["date"] = all_logs["date"].astype(int)
    all_logs["time_ms"] = all_logs["time_ms"].astype(np.int64)
    if args.debug:
        if split_protocol == "random_aux_split":
            debug_parts = [
                all_logs[(all_logs["is_random"] == 0) & (all_logs["date"] <= 20220426)].head(5000),
                all_logs[(all_logs["is_random"] == 0) & (all_logs["date"] >= 20220427) & (all_logs["date"] <= 20220430)].head(5000),
                all_logs[(all_logs["is_random"] == 0) & (all_logs["date"] >= 20220501)].head(5000),
                all_logs[(all_logs["is_random"] == 1) & (all_logs["date"] >= 20220422) & (all_logs["date"] <= 20220426)].head(5000),
                all_logs[(all_logs["is_random"] == 1) & (all_logs["date"] >= 20220427) & (all_logs["date"] <= 20220430)].head(5000),
                all_logs[(all_logs["is_random"] == 1) & (all_logs["date"] >= 20220501)].head(5000),
            ]
        else:
            debug_parts = [
                all_logs[(all_logs["is_random"] == 0) & (all_logs["date"] <= 20220421)].head(5000),
                all_logs[(all_logs["is_random"] == 0) & (all_logs["date"] >= 20220422) & (all_logs["date"] <= 20220430)].head(5000),
                all_logs[(all_logs["is_random"] == 0) & (all_logs["date"] >= 20220501)].head(5000),
            ]
        all_logs = pd.concat(debug_parts, ignore_index=True)
    if use_time_context:
        all_logs = add_time_context_features(all_logs)
    else:
        all_logs["hour_of_day"] = 0
        all_logs["day_of_week"] = 0
        all_logs["is_weekend"] = 0
    all_logs, position_summary = add_position_features(all_logs)

    for c in ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view"]:
        all_logs[c] = all_logs[c].fillna(0).astype(np.int8)

    user_df = pd.read_csv(user_path)
    video_basic_df = pd.read_csv(video_basic_path)

    all_logs["user_id_raw"] = all_logs["user_id"].astype(str)
    all_logs["video_id_raw"] = all_logs["video_id"].astype(str)
    user_df["user_id_raw"] = user_df["user_id"].astype(str)
    video_basic_df["video_id_raw"] = video_basic_df["video_id"].astype(str)
    author_source_col = next((c for c in AUTHOR_FIELD_CANDIDATES if c in video_basic_df.columns), None)
    if author_source_col is not None:
        video_basic_df["author_id_raw"] = video_basic_df[author_source_col].fillna("<unk>").astype(str)
    else:
        video_basic_df["author_id_raw"] = "<unk>"

    df = all_logs.merge(user_df.drop(columns=["user_id"]), on="user_id_raw", how="left")
    df = df.merge(video_basic_df.drop(columns=["video_id"]), on="video_id_raw", how="left")

    video_stat_source_cols = []
    if use_video_stat:
        video_stat_path = resolve_raw_file(data_raw_dir, "video_features_statistic_pure.csv")
        video_stat_df = pd.read_csv(video_stat_path)
        video_stat_source_cols = [c for c in video_stat_df.columns if c != "video_id"]
        video_stat_df["video_id_raw"] = video_stat_df["video_id"].astype(str)
        df = df.merge(video_stat_df.drop(columns=["video_id"]), on="video_id_raw", how="left")

    cat_path = data_raw_dir / "kuairand_video_categories.csv"
    if cat_path.exists():
        cat_df = load_categories_for_videos(cat_path, set(df["video_id_raw"].unique().tolist()))
        cat_df = cat_df.rename(
            columns={
                "final_video_id": "video_id_raw",
                "first_level_category_id": "category_l1_raw",
                "second_level_category_id": "category_l2_raw",
                "third_level_category_id": "category_l3_raw",
                "fourth_level_category_id": "category_l4_raw",
            }
        )
        df = df.merge(cat_df, on="video_id_raw", how="left")
    else:
        df["category_l1_raw"] = np.nan
        df["category_l2_raw"] = np.nan
        df["category_l3_raw"] = np.nan
        df["category_l4_raw"] = np.nan

    caption_svd_cols = []
    if use_caption:
        cap_path = data_raw_dir / "kuairand_video_captions.csv"
        if cap_path.exists():
            caps = []
            use_ids = set(df["video_id_raw"].unique().tolist())
            for chunk in pd.read_csv(cap_path, usecols=["final_video_id", "caption"], chunksize=500_000):
                chunk["final_video_id"] = chunk["final_video_id"].astype(str)
                sub = chunk[chunk["final_video_id"].isin(use_ids)]
                if len(sub) > 0:
                    caps.append(sub)
            if caps:
                caption_df = pd.concat(caps, ignore_index=True).drop_duplicates("final_video_id")
                caption_df["caption"] = caption_df["caption"].fillna("")
                tfidf = TfidfVectorizer(max_features=20_000)
                X = tfidf.fit_transform(caption_df["caption"].values)
                svd = TruncatedSVD(n_components=64, random_state=int(config.get("seed", 2025)))
                X_svd = svd.fit_transform(X)
                svd_cols = [f"caption_svd_{i}" for i in range(X_svd.shape[1])]
                caption_svd_cols = svd_cols
                svd_df = pd.DataFrame(X_svd, columns=svd_cols)
                svd_df["video_id_raw"] = caption_df["final_video_id"].values
                df = df.merge(svd_df, on="video_id_raw", how="left")
                for c in svd_cols:
                    df[c] = df[c].fillna(0.0)

    # Heavy-tail numeric transformations: log1p + clip p99 + bucketize
    heavy_cols = [
        "video_duration",
        "register_days",
        "fans_user_num",
        "follow_user_num",
        "friend_user_num",
    ]
    bucket_edges: Dict[str, list[float]] = {}
    for c in heavy_cols:
        if c in df.columns:
            edges = fit_log_bucket_edges(df[c], num_bins=20)
            bucket_edges[f"{c}_bucket"] = edges
            df[f"{c}_bucket"] = apply_log_bucket(df[c], edges)
        else:
            bucket_edges[f"{c}_bucket"] = [0.0, 1.0]
            df[f"{c}_bucket"] = 1

    # Play ratio and its bucket
    denom = np.maximum(pd.to_numeric(df["duration_ms"], errors="coerce").fillna(1.0).values, 1.0)
    numer = pd.to_numeric(df["play_time_ms"], errors="coerce").fillna(0.0).values
    play_ratio = np.clip(numer / denom, 0.0, 5.0)
    play_ratio_bounds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    df["play_ratio_bucket"] = bucketize_fixed(play_ratio, play_ratio_bounds)
    bucket_edges["play_ratio_bucket"] = play_ratio_bounds

    # Tab mapping: keep high-frequency tab 0/1/2/4, others -> other_tab
    high_tabs = {"0", "1", "2", "4"}
    tab_raw = df["tab"].fillna("other_tab").astype(str)
    tab_norm = tab_raw.where(tab_raw.isin(high_tabs), "other_tab")
    tab_mapping = {"0": 1, "1": 2, "2": 3, "4": 4, "other_tab": 5}
    df["tab"] = tab_norm.map(tab_mapping).fillna(0).astype(np.int64)

    sparse_sources = {
        "user_id": df["user_id_raw"],
        "video_id": df["video_id_raw"],
        "category_l1_id": df.get("category_l1_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        "category_l2_id": df.get("category_l2_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        "category_l3_id": df.get("category_l3_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        "category_l4_id": df.get("category_l4_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        "video_type": df.get("video_type", pd.Series(["<unk>"] * len(df), index=df.index)),
        "upload_type": df.get("upload_type", pd.Series(["<unk>"] * len(df), index=df.index)),
        "music_type": df.get("music_type", pd.Series(["<unk>"] * len(df), index=df.index)),
        "tag": df.get("tag", pd.Series(["<unk>"] * len(df), index=df.index)),
        "user_active_degree": df.get("user_active_degree", pd.Series(["<unk>"] * len(df), index=df.index)),
        "author_id": df.get("author_id_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
    }
    for i in range(18):
        col = f"onehot_feat{i}"
        if col in df.columns:
            sparse_sources[col] = df[col].fillna("<unk>").astype(str)

    sparse_maps: Dict[str, Dict[Any, int]] = {}
    vocab_sizes: Dict[str, int] = {}

    for name, series in sparse_sources.items():
        mapping = fit_sparse_mapping(series)
        sparse_maps[name] = mapping
        vocab_sizes[name] = len(mapping) + 1

    vocab_sizes["tab"] = max(tab_mapping.values()) + 1
    sparse_maps["tab"] = tab_mapping

    # Apply sparse mappings
    df["user_id"] = apply_sparse_mapping(df["user_id_raw"], sparse_maps["user_id"])
    df["target_video_id"] = apply_sparse_mapping(df["video_id_raw"], sparse_maps["video_id"])
    df["target_category_l1"] = apply_sparse_mapping(
        df.get("category_l1_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["category_l1_id"],
    )
    df["target_category_l2"] = apply_sparse_mapping(
        df.get("category_l2_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["category_l2_id"],
    )
    df["target_category_l3"] = apply_sparse_mapping(
        df.get("category_l3_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["category_l3_id"],
    )
    df["target_category_l4"] = apply_sparse_mapping(
        df.get("category_l4_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["category_l4_id"],
    )
    df["target_video_type"] = apply_sparse_mapping(
        df.get("video_type", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["video_type"],
    )
    df["target_tag"] = apply_sparse_mapping(
        df.get("tag", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["tag"],
    )
    df["user_active_degree"] = apply_sparse_mapping(
        df.get("user_active_degree", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["user_active_degree"],
    )
    df["target_author_id"] = apply_sparse_mapping(
        df.get("author_id_raw", pd.Series(["<unk>"] * len(df), index=df.index)),
        sparse_maps["author_id"],
    )
    for i in range(18):
        col = f"onehot_feat{i}"
        if col in sparse_maps:
            df[col] = apply_sparse_mapping(df[col].fillna("<unk>").astype(str), sparse_maps[col])

    # User heavy-tail buckets used by static MBC branch
    df["register_days_bucket"] = df["register_days_bucket"].fillna(1).astype(np.int64)
    df["fans_user_num_bucket"] = df["fans_user_num_bucket"].fillna(1).astype(np.int64)
    df["follow_user_num_bucket"] = df["follow_user_num_bucket"].fillna(1).astype(np.int64)
    df["friend_user_num_bucket"] = df["friend_user_num_bucket"].fillna(1).astype(np.int64)

    # Duration bucket used by model
    df["target_duration_bucket"] = df["video_duration_bucket"].fillna(1).astype(np.int64)
    dense_cols = []
    if use_dense_features and not use_dense_history_only:
        dense_cols.extend(
            add_train_target_priors(
                df,
                train_end_date=int(config.get("train_end_date", 20220421)),
                alpha=float(config.get("prior_alpha", 20.0)),
            )
        )
    if use_video_stat:
        dense_cols.extend(add_video_stat_dense_features(df, video_stat_source_cols))
    if use_caption:
        dense_cols.extend(add_caption_dense_features(df, caption_svd_cols))
    author_prior_cols = []
    if use_author_prior:
        author_prior_cols = add_author_priors(
            df,
            train_end_date=int(config.get("author_prior_train_end_date", config.get("train_end_date", 20220426))),
            alpha=float(config.get("author_prior_alpha", config.get("prior_alpha", 20.0))),
        )

    bucket_sizes = {
        "duration_bucket": int(df["target_duration_bucket"].max()),
        "play_ratio_bucket": int(df["play_ratio_bucket"].max()),
        "time_gap_bucket": 12,
        "register_days_bucket": int(df["register_days_bucket"].max()),
        "fans_user_num_bucket": int(df["fans_user_num_bucket"].max()),
        "follow_user_num_bucket": int(df["follow_user_num_bucket"].max()),
        "friend_user_num_bucket": int(df["friend_user_num_bucket"].max()),
    }
    vocab_sizes["duration_bucket"] = bucket_sizes["duration_bucket"] + 1
    vocab_sizes["register_days_bucket"] = bucket_sizes["register_days_bucket"] + 1
    vocab_sizes["fans_user_num_bucket"] = bucket_sizes["fans_user_num_bucket"] + 1
    vocab_sizes["follow_user_num_bucket"] = bucket_sizes["follow_user_num_bucket"] + 1
    vocab_sizes["friend_user_num_bucket"] = bucket_sizes["friend_user_num_bucket"] + 1
    vocab_sizes["hour_of_day"] = 25
    vocab_sizes["day_of_week"] = 8
    vocab_sizes["is_weekend"] = 3
    bucket_sizes["position_bucket"] = int(df["position_bucket"].max()) if "position_bucket" in df.columns and len(df) > 0 else 0

    df = df.sort_values(["user_id_raw", "time_ms"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    build_history_dense = bool(use_history_dense_features or use_dense_history_only)
    df = build_histories(
        df,
        max_seq_len=max_seq_len,
        use_dense_features=build_history_dense,
        history_mode=history_mode,
    )

    model_cols = [
        "user_id",
        "target_video_id",
        "label",
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
        "hist_video_id",
        "hist_category_l1",
        "hist_category_l2",
        "hist_category_l3",
        "hist_category_l4",
        "hist_duration_bucket",
        "hist_tab",
        "hist_author_id",
        "hist_action_vector",
        "hist_play_ratio_bucket",
        "hist_time_gap_bucket",
        "hist_mask",
        "date",
    ]
    model_cols.extend(["is_random", "hour_of_day", "day_of_week", "is_weekend", "display_position", "position_bucket", "target_author_id"])
    for i in range(18):
        col = f"onehot_feat{i}"
        if col in df.columns:
            model_cols.append(col)
    history_dense_cols = HISTORY_DENSE_COLS if build_history_dense else []
    if dense_cols:
        model_cols.extend(dense_cols)
    if history_dense_cols:
        model_cols.extend(history_dense_cols)
    if author_prior_cols:
        model_cols.extend(author_prior_cols)
    semantic_cols = []
    if use_video_semantic_emb:
        for i in range(int(config.get("semantic_proj_input_dim", 64))):
            col = f"target_semantic_emb_{i}"
            df[col] = 0.0
            semantic_cols.append(col)
            model_cols.append(col)
    semantic_match_cols = []
    if use_semantic_match_features:
        for i in range(int(config.get("semantic_match_feature_dim", 8))):
            col = f"semantic_match_{i}"
            df[col] = 0.0
            semantic_match_cols.append(col)
            model_cols.append(col)
    df_model = df[model_cols].copy()

    splits = split_frames(df_model, split_protocol)
    full_train_df = splits["train"]
    full_valid_df = splits["valid"]
    full_test_df = splits["test"]
    random_aux_train_df = splits["random_aux_train"]
    random_valid_df = splits["random_valid"]
    random_test_df = splits["random_test"]
    split_stats = {name: summarize_split(frame) for name, frame in splits.items()}

    debug_train_df = full_train_df.head(5000).copy()
    debug_valid_df = full_valid_df.head(5000).copy()
    debug_test_df = full_test_df.head(5000).copy()
    debug_random_aux_train_df = random_aux_train_df.head(5000).copy()
    debug_random_valid_df = random_valid_df.head(5000).copy()

    if not args.debug:
        save_dataframe(full_train_df.drop(columns=["date"]), processed_dir / "train.parquet")
        save_dataframe(full_valid_df.drop(columns=["date"]), processed_dir / "valid.parquet")
        save_dataframe(full_test_df.drop(columns=["date"]), processed_dir / "test.parquet")
        if split_protocol == "random_aux_split":
            save_dataframe(random_aux_train_df.drop(columns=["date"]), processed_dir / "random_aux_train.parquet")
            save_dataframe(random_valid_df.drop(columns=["date"]), processed_dir / "random_valid.parquet")

    save_dataframe(debug_train_df.drop(columns=["date"]), processed_dir / "train_debug.parquet")
    save_dataframe(debug_valid_df.drop(columns=["date"]), processed_dir / "valid_debug.parquet")
    save_dataframe(debug_test_df.drop(columns=["date"]), processed_dir / "test_debug.parquet")
    if split_protocol == "random_aux_split":
        save_dataframe(debug_random_aux_train_df.drop(columns=["date"]), processed_dir / "random_aux_train_debug.parquet")
        save_dataframe(debug_random_valid_df.drop(columns=["date"]), processed_dir / "random_valid_debug.parquet")

    feature_maps = {
        "sparse_maps": sparse_maps,
        "vocab_sizes": vocab_sizes,
        "bucket_edges": bucket_edges,
        "bucket_sizes": bucket_sizes,
        "max_seq_len": max_seq_len,
        "dense_cols": dense_cols,
        "dense_dim": len(dense_cols),
        "history_dense_cols": history_dense_cols,
        "history_dense_dim": len(history_dense_cols),
        "author_prior_cols": author_prior_cols,
        "author_prior_dim": len(author_prior_cols),
        "calibration_dense_cols": dense_cols,
        "calibration_dense_dim": len(dense_cols),
        "semantic_cols": semantic_cols,
        "semantic_dim": len(semantic_cols),
        "semantic_match_cols": semantic_match_cols,
        "semantic_match_dim": len(semantic_match_cols),
        "history_mode": history_mode,
        "split_protocol": split_protocol,
    }
    save_pickle(feature_maps, processed_dir / "feature_maps.pkl")

    summary = {
        "mode": "debug" if args.debug else "full",
        "split_protocol": split_protocol,
        "standard_train_count": int(len(full_train_df)),
        "standard_valid_count": int(len(full_valid_df)),
        "standard_test_count": int(len(full_test_df)),
        "random_aux_train_count": int(len(random_aux_train_df)),
        "random_valid_count": int(len(random_valid_df)),
        "random_test_count": int(len(random_test_df)),
        "full_train_rows": int(len(full_train_df)),
        "full_valid_rows": int(len(full_valid_df)),
        "full_test_rows": int(len(full_test_df)),
        "debug_train_rows": int(len(debug_train_df)),
        "debug_valid_rows": int(len(debug_valid_df)),
        "debug_test_rows": int(len(debug_test_df)),
        "debug_random_aux_train_rows": int(len(debug_random_aux_train_df)),
        "max_seq_len": max_seq_len,
        "history_mode": history_mode,
        "use_video_stat": use_video_stat,
        "use_caption": use_caption,
        "use_time_context": use_time_context,
        "use_dense_features": use_dense_features,
        "use_dense_history_only": use_dense_history_only,
        "use_history_dense_features": use_history_dense_features,
        "use_author_features": use_author_features,
        "use_author_prior": use_author_prior,
        "author_source_col": author_source_col,
        "position_summary": position_summary,
        "dense_dim": len(dense_cols),
        "dense_cols": dense_cols,
        "history_dense_dim": len(history_dense_cols),
        "history_dense_cols": history_dense_cols,
        "author_prior_dim": len(author_prior_cols),
        "author_prior_cols": author_prior_cols,
        "semantic_dim": len(semantic_cols),
        "semantic_cols": semantic_cols,
        "semantic_match_dim": len(semantic_match_cols),
        "semantic_match_cols": semantic_match_cols,
        "dense_feature_names": dense_cols + history_dense_cols + author_prior_cols,
        "split_stats": split_stats,
        "hist_len_stats": {"train": split_stats["train"], "valid": split_stats["valid"], "test": split_stats["test"]},
        "bucket_sizes": bucket_sizes,
        "vocab_sizes": vocab_sizes,
        "standard_train_date_range": "<=20220426" if split_protocol == "random_aux_split" else "<=20220421",
        "random_aux_train_date_range": "20220422~20220426" if split_protocol == "random_aux_split" else "none",
        "standard_valid_date_range": "20220427~20220430" if split_protocol == "random_aux_split" else "20220422~20220430",
        "standard_test_date_range": ">=20220501",
    }
    save_json(summary, processed_dir / "preprocess_summary.json")

    print("Preprocess done.")
    print(summary)


if __name__ == "__main__":
    main()
