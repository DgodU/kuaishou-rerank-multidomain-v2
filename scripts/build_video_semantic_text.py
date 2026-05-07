from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, save_dataframe


PROMPT_TEMPLATE = """任务：为短视频点击率预估和个性化重排生成视频语义表示。
请关注视频主题、内容类型、实体、情绪风格、用户兴趣点和消费场景。

视频信息：
标题/描述：{caption}
封面文字：{show_cover_text}
一级类目：{category_l1}
二级类目：{category_l2}
三级类目：{category_l3}
四级类目：{category_l4}
视频时长：{duration}
"""


def _read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", **kwargs)


def _resolve_raw_file(data_raw_dir: Path, filename: str) -> Path:
    for path in [data_raw_dir / filename, data_raw_dir / "KuaiRand-Pure" / "data" / filename]:
        if path.exists():
            return path
    return data_raw_dir / filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/semantic/video_semantic_text.parquet")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_raw_dir = PROJECT_ROOT / "data" / "raw"
    captions_path = data_raw_dir / "kuairand_video_captions.csv"
    categories_path = data_raw_dir / "kuairand_video_categories.csv"
    basic_path = _resolve_raw_file(data_raw_dir, "video_features_basic_pure.csv")

    basic = _read_csv_if_exists(basic_path)
    if basic.empty or "video_id" not in basic.columns:
        raise FileNotFoundError(f"Cannot build semantic text without video_features_basic_pure.csv: {basic_path}")
    basic = basic.copy()
    basic["video_id"] = pd.to_numeric(basic["video_id"], errors="coerce").fillna(0).astype("int64")
    basic["video_id_raw"] = basic["video_id"].astype(str)
    if args.debug:
        basic = basic.head(1000).copy()
    use_ids = set(basic["video_id_raw"].tolist())

    captions = _read_csv_if_exists(captions_path)
    if not captions.empty and "final_video_id" in captions.columns:
        captions = captions.copy()
        captions["video_id_raw"] = captions["final_video_id"].astype(str)
        keep_cols = [c for c in ["video_id_raw", "caption", "show_cover_text"] if c in captions.columns]
        captions = captions.loc[captions["video_id_raw"].isin(use_ids), keep_cols].drop_duplicates("video_id_raw")
    else:
        captions = pd.DataFrame(columns=["video_id_raw", "caption", "show_cover_text"])

    categories = _read_csv_if_exists(categories_path)
    if not categories.empty and "final_video_id" in categories.columns:
        categories = categories.copy()
        categories["video_id_raw"] = categories["final_video_id"].astype(str)
        rename_map = {
            "first_level_category_id": "category_l1",
            "second_level_category_id": "category_l2",
            "third_level_category_id": "category_l3",
            "fourth_level_category_id": "category_l4",
        }
        categories = categories.rename(columns=rename_map)
        keep_cols = [c for c in ["video_id_raw", "category_l1", "category_l2", "category_l3", "category_l4"] if c in categories.columns]
        categories = categories.loc[categories["video_id_raw"].isin(use_ids), keep_cols].drop_duplicates("video_id_raw")
    else:
        categories = pd.DataFrame(columns=["video_id_raw", "category_l1", "category_l2", "category_l3", "category_l4"])

    df = basic[["video_id", "video_id_raw"]].copy()
    for candidate in ["video_duration", "duration_ms", "duration"]:
        if candidate in basic.columns:
            df["duration"] = basic[candidate]
            break
    if "duration" not in df.columns:
        df["duration"] = ""
    df = df.merge(captions, on="video_id_raw", how="left")
    df = df.merge(categories, on="video_id_raw", how="left")

    for col in ["caption", "show_cover_text", "category_l1", "category_l2", "category_l3", "category_l4", "duration"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["caption_missing"] = (df["caption"].str.len() == 0).astype("int8")
    df["cover_text_missing"] = (df["show_cover_text"].str.len() == 0).astype("int8")
    df["category_missing"] = ((df["category_l1"].str.len() == 0) & (df["category_l2"].str.len() == 0)).astype("int8")
    df["duration_missing"] = (df["duration"].str.len() == 0).astype("int8")
    df["semantic_text"] = [
        PROMPT_TEMPLATE.format(
            caption=row.caption,
            show_cover_text=row.show_cover_text,
            category_l1=row.category_l1,
            category_l2=row.category_l2,
            category_l3=row.category_l3,
            category_l4=row.category_l4,
            duration=row.duration,
        )
        for row in df.itertuples(index=False)
    ]
    out = df[["video_id", "semantic_text", "caption_missing", "cover_text_missing", "category_missing", "duration_missing"]]
    output_path = PROJECT_ROOT / args.output
    ensure_dir(output_path.parent)
    saved = save_dataframe(out, output_path)
    print(f"Saved semantic text: {saved} | rows={len(out)}")
    if args.debug:
        for row in out.head(3).itertuples(index=False):
            print(f"SAMPLE video_id={row.video_id}\n{row.semantic_text[:1000]}")


if __name__ == "__main__":
    main()
