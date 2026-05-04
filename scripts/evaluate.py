from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import KuaiRandDataset, kuairand_collate_fn
from src.models.ads import ADSModel
from src.models.ads_transformer_side import ADSTransformerSideModel
from src.models.ads_transformer_side_mbc import ADSTransformerSideMBCModel
from src.training.trainer import CTRTrainer
from src.utils.io import load_pickle, load_yaml
from src.utils.logger import get_logger


def build_model(model_name: str, config: dict, feature_maps: dict) -> torch.nn.Module:
    if model_name == "ads":
        return ADSModel(config, feature_maps)
    if model_name == "ads_transformer_side":
        return ADSTransformerSideModel(config, feature_maps)
    if model_name == "ads_transformer_side_mbc":
        return ADSTransformerSideMBCModel(config, feature_maps)
    raise ValueError(f"Unknown model_name: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_dir = PROJECT_ROOT / config.get("data_dir", "data/processed")

    split_file = f"{args.split}_debug.parquet" if args.debug else f"{args.split}.parquet"
    dataset = KuaiRandDataset(data_dir / split_file)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 2048)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        collate_fn=kuairand_collate_fn,
        pin_memory=True,
    )

    feature_maps = load_pickle(data_dir / "feature_maps.pkl")
    model = build_model(config["model_name"], config, feature_maps)

    device = torch.device("cuda" if (config.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    logger = get_logger("evaluate")
    trainer = CTRTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        config=config,
        logger=logger,
    )
    metrics = trainer.evaluate(loader)
    logger.info(
        "Split=%s | auc=%.6f | gauc=%.6f | logloss=%.6f",
        args.split,
        metrics["auc"],
        metrics["gauc"],
        metrics["logloss"],
    )


if __name__ == "__main__":
    main()
