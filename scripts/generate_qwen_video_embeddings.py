from __future__ import annotations

import argparse
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, load_dataframe, save_dataframe


def _mock_embedding(video_id: int, text: str, dim: int) -> list[float]:
    key = f"{video_id}|{text}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little", signed=False) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(float).tolist()


def _call_qwen_embedding_api(texts: list[str], model_name: str) -> list[list[float]]:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set DASHSCOPE_API_KEY/QWEN_API_KEY or use --mock_debug. "
            "This script is only for offline embedding generation; training never calls APIs."
        )
    try:
        from dashscope import TextEmbedding
    except Exception as exc:
        raise RuntimeError("dashscope package is required for real Qwen embedding generation.") from exc
    resp = TextEmbedding.call(model=model_name, input=texts, api_key=api_key)
    if getattr(resp, "status_code", None) != 200:
        raise RuntimeError(f"Qwen embedding API failed: {resp}")
    embeddings = []
    for item in resp.output.get("embeddings", []):
        embeddings.append([float(x) for x in item["embedding"]])
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding API returned {len(embeddings)} embeddings for {len(texts)} texts")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/semantic/video_semantic_text.parquet")
    parser.add_argument("--output", type=str, default="data/semantic/video_semantic_emb.parquet")
    parser.add_argument("--model_name", type=str, default="text-embedding-v3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock_debug", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mock_dim", type=int, default=128)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    df = load_dataframe(input_path)
    required = {"video_id", "semantic_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input semantic text missing columns: {sorted(missing)}")
    if args.debug:
        df = df.head(1000).copy()

    done = pd.DataFrame()
    done_ids: set[int] = set()
    if args.resume and (output_path.exists() or output_path.with_suffix(".pkl").exists()):
        done = load_dataframe(output_path)
        if "video_id" in done.columns:
            done_ids = set(pd.to_numeric(done["video_id"], errors="coerce").fillna(0).astype("int64").tolist())

    rows = []
    created_at = datetime.now(timezone.utc).isoformat()
    pending = df[~pd.to_numeric(df["video_id"], errors="coerce").fillna(0).astype("int64").isin(done_ids)].copy()
    for start in range(0, len(pending), max(1, args.batch_size)):
        batch = pending.iloc[start : start + max(1, args.batch_size)]
        video_ids = pd.to_numeric(batch["video_id"], errors="coerce").fillna(0).astype("int64").tolist()
        texts = batch["semantic_text"].fillna("").astype(str).tolist()
        if args.mock_debug:
            embeddings = [_mock_embedding(int(vid), text, int(args.mock_dim)) for vid, text in zip(video_ids, texts)]
            model_name = f"mock_debug_dim{int(args.mock_dim)}"
        else:
            embeddings = _call_qwen_embedding_api(texts, args.model_name)
            model_name = args.model_name
        for vid, emb in zip(video_ids, embeddings):
            rows.append(
                {
                    "video_id": int(vid),
                    "semantic_emb": emb,
                    "semantic_missing_flag": int(len(emb) == 0),
                    "embedding_model": model_name,
                    "created_at": created_at,
                }
            )
        if args.sleep > 0:
            time.sleep(float(args.sleep))
        if rows and args.resume:
            current = pd.DataFrame(rows)
            out = pd.concat([done, current], ignore_index=True) if len(done) else current
            ensure_dir(output_path.parent)
            save_dataframe(out.drop_duplicates("video_id"), output_path)

    new_df = pd.DataFrame(rows)
    out = pd.concat([done, new_df], ignore_index=True) if len(done) else new_df
    if len(out) == 0:
        out = done
    ensure_dir(output_path.parent)
    saved = save_dataframe(out.drop_duplicates("video_id"), output_path)
    print(f"Saved semantic embeddings: {saved} | rows={len(out.drop_duplicates('video_id'))} | mock_debug={args.mock_debug}")


if __name__ == "__main__":
    main()
