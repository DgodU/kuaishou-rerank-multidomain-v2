import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_dataframe(df: pd.DataFrame, parquet_path: str | Path) -> Path:
    target = Path(parquet_path)
    ensure_dir(target.parent)
    try:
        df.to_parquet(target, index=False)
        return target
    except Exception:
        pkl_path = target.with_suffix(".pkl")
        df.to_pickle(pkl_path)
        return pkl_path


def load_dataframe(parquet_path: str | Path) -> pd.DataFrame:
    target = Path(parquet_path)
    if target.exists():
        try:
            return pd.read_parquet(target)
        except Exception:
            pass
    pkl_path = target.with_suffix(".pkl")
    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    raise FileNotFoundError(f"Neither {target} nor {pkl_path} exists")
