from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.utils.io import load_pickle


@dataclass
class FeatureMaps:
    sparse_maps: Dict[str, Dict[Any, int]]
    vocab_sizes: Dict[str, int]
    bucket_edges: Dict[str, list]
    bucket_sizes: Dict[str, int]

    @classmethod
    def from_pickle(cls, path: str | Path) -> "FeatureMaps":
        data = load_pickle(path)
        return cls(
            sparse_maps=data.get("sparse_maps", {}),
            vocab_sizes=data.get("vocab_sizes", {}),
            bucket_edges=data.get("bucket_edges", {}),
            bucket_sizes=data.get("bucket_sizes", {}),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMaps":
        return cls(
            sparse_maps=data.get("sparse_maps", {}),
            vocab_sizes=data.get("vocab_sizes", {}),
            bucket_edges=data.get("bucket_edges", {}),
            bucket_sizes=data.get("bucket_sizes", {}),
        )
