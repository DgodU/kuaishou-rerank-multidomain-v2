from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, NamedTuple

import numpy as np
import pandas as pd

from src.utils.io import load_dataframe


class SemanticEmbeddingBundle(NamedTuple):
    video_id_to_index: Dict[int, int]
    semantic_matrix: np.ndarray
    missing_flags: np.ndarray | None
    raw_dim: int


def _parse_embedding_value(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        try:
            parsed = ast.literal_eval(text)
            return np.asarray(parsed, dtype=np.float32).reshape(-1)
        except Exception:
            return np.fromstring(text.strip("[]"), sep=",", dtype=np.float32).reshape(-1)
    return np.zeros(0, dtype=np.float32)


def load_video_semantic_embeddings(path: str | Path) -> SemanticEmbeddingBundle:
    target = Path(path)
    df = load_dataframe(target)
    if "video_id" not in df.columns:
        raise ValueError(f"semantic embedding file must contain video_id: {target}")

    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda x: int(x.split("_", 1)[1]) if x.split("_", 1)[1].isdigit() else x,
    )
    nan_count = 0
    if emb_cols:
        values = df[emb_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        nan_count = int(np.isnan(values).sum() + np.isinf(values).sum())
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    elif "semantic_emb" in df.columns:
        parsed = [_parse_embedding_value(x) for x in df["semantic_emb"].to_numpy()]
        raw_dim = max((int(x.size) for x in parsed), default=0)
        values = np.zeros((len(parsed), raw_dim), dtype=np.float32)
        for i, arr in enumerate(parsed):
            use_dim = min(raw_dim, int(arr.size))
            if use_dim > 0:
                values[i, :use_dim] = arr[:use_dim]
        nan_count = int(np.isnan(values).sum() + np.isinf(values).sum())
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        raise ValueError(f"semantic embedding file must contain semantic_emb or emb_0..emb_d columns: {target}")

    raw_dim = int(values.shape[1]) if values.ndim == 2 else 0
    if raw_dim <= 0:
        raise ValueError(f"semantic embedding dim is zero: {target}")

    norms = np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)
    nonzero = norms.squeeze(-1) > 0
    values = np.divide(values, np.maximum(norms, 1e-12), out=np.zeros_like(values), where=norms > 0)

    video_ids = pd.to_numeric(df["video_id"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    missing_flags = None
    if "semantic_missing_flag" in df.columns:
        missing_flags = pd.to_numeric(df["semantic_missing_flag"], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    missing_count = int((~nonzero).sum()) if missing_flags is None else int(np.asarray(missing_flags > 0).sum())

    matrix = np.zeros((len(values) + 1, raw_dim), dtype=np.float32)
    matrix[1:] = values
    video_id_to_index: Dict[int, int] = {}
    for i, vid in enumerate(video_ids, start=1):
        if int(vid) > 0 and int(vid) not in video_id_to_index:
            video_id_to_index[int(vid)] = i

    norm_values = np.linalg.norm(matrix[1:], axis=1) if len(matrix) > 1 else np.zeros(0, dtype=np.float32)
    norm_stats = {
        "min": float(norm_values.min()) if norm_values.size else 0.0,
        "mean": float(norm_values.mean()) if norm_values.size else 0.0,
        "max": float(norm_values.max()) if norm_values.size else 0.0,
    }
    print(
        "Loaded video semantic embeddings | "
        f"number_of_semantic_videos={len(video_id_to_index)} | raw_dim={raw_dim} | "
        f"nan_count={nan_count} | missing_count={missing_count} | norm_statistics={norm_stats}"
    )
    return SemanticEmbeddingBundle(video_id_to_index, matrix, missing_flags, raw_dim)
