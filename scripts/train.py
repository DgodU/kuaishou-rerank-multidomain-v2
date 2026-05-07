from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import KuaiRandDataset, kuairand_collate_fn
from src.data.samplers import UserGroupBatchSampler
from src.models import ADSModel, ADSTransformerSideModel, ADSTransformerSideMBCModel
from src.training.trainer import CTRTrainer
from src.utils.io import ensure_dir, load_pickle, load_yaml, save_json
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def build_model(config: dict, feature_maps: dict) -> torch.nn.Module:
    model_name = config["model_name"]
    if model_name == "ads":
        return ADSModel(config, feature_maps)
    if model_name == "ads_transformer_side":
        return ADSTransformerSideModel(config, feature_maps)
    if model_name == "ads_transformer_side_mbc":
        return ADSTransformerSideMBCModel(config, feature_maps)
    raise ValueError(f"Unknown model_name: {model_name}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def log_batch_shape_summary(logger, batch: Dict[str, torch.Tensor]) -> None:
    parts = []
    for k, v in batch.items():
        parts.append(f"{k}:{tuple(v.shape)}")
    logger.info("Batch tensor shape summary | %s", " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.debug:
        config["batch_size"] = min(int(config.get("batch_size", 2048)), 256)
        config["epochs"] = 1
        config["num_workers"] = 0
        config["_debug_shapes"] = True
    set_seed(int(config.get("seed", 2025)))

    data_dir = PROJECT_ROOT / config.get("data_dir", "data/processed")
    ckpt_dir = ensure_dir(PROJECT_ROOT / config.get("checkpoint_dir", "checkpoints"))
    log_dir = ensure_dir(PROJECT_ROOT / config.get("log_dir", "logs"))
    out_dir = ensure_dir(PROJECT_ROOT / config.get("output_dir", "outputs"))
    experiment_name = str(config.get("experiment_name", config.get("run_name", ""))).strip()
    run_name = experiment_name
    run_suffix = f"_{run_name}" if run_name else ""

    mode_name = "debug" if args.debug else "full"
    log_base_name = f"{config['model_name']}{run_suffix}"
    logger = get_logger(
        name=f"train.{log_base_name}.{mode_name}",
        log_file=log_dir / (f"{log_base_name}.log" if mode_name == "full" else f"{log_base_name}_debug.log"),
    )
    logger.info(
        "Run config | config_path=%s | experiment_name=%s | split_protocol=%s | data_dir=%s | history_mode=%s | model_name=%s | attention_type=%s | use_side_bias=%s | use_time_context=%s | use_pcrg_token=%s | use_transformer_fusion=%s | use_mbc_slices=%s | use_dense_history_only=%s | use_random_aux=%s | use_position_bias_tower=%s | use_rank_calib_split=%s | use_history_dense_features=%s | seed=%s",
        args.config,
        experiment_name or log_base_name,
        config.get("split_protocol", "original"),
        data_dir,
        config.get("history_mode", "all"),
        config.get("model_name"),
        config.get("attention_type", "dot"),
        config.get("use_side_bias", config.get("use_side_attention_bias", False)),
        config.get("use_time_context", False),
        config.get("use_pcrg_token", False),
        config.get("use_transformer_fusion", False),
        config.get("use_mbc_slices", False),
        config.get("use_dense_history_only", False),
        config.get("use_random_aux", False),
        config.get("use_position_bias_tower", False),
        config.get("use_rank_calib_split", False),
        config.get("use_history_dense_features", False),
        config.get("seed", 2025),
    )
    logger.info(
        "Semantic config | use_video_semantic_emb=%s | use_simtier_features=%s | use_semantic_long_short=%s | semantic_inject=%s | semantic_emb_path=%s",
        config.get("use_video_semantic_emb", False),
        config.get("use_simtier_features", False),
        config.get("use_semantic_long_short", False),
        config.get("semantic_inject", ""),
        config.get("semantic_emb_path", ""),
    )

    train_file = data_dir / ("train_debug.parquet" if args.debug else "train.parquet")
    valid_file = data_dir / ("valid_debug.parquet" if args.debug else "valid.parquet")
    test_file = data_dir / ("test_debug.parquet" if args.debug else "test.parquet")

    feature_maps = load_pickle(data_dir / "feature_maps.pkl")
    logger.info("Loading datasets: %s | %s", train_file, valid_file)
    train_ds = KuaiRandDataset(train_file, config=config, feature_maps=feature_maps)
    valid_ds = KuaiRandDataset(valid_file, config=config, feature_maps=feature_maps)
    logger.info("Loaded train/valid dataset sizes: %d / %d", len(train_ds), len(valid_ds))

    num_workers = int(config.get("num_workers", 4))
    batch_size = int(config.get("batch_size", 2048))
    pin_memory = bool(config.get("pin_memory", True))
    persistent_workers = bool(config.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(config.get("prefetch_factor", 2))

    common_loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": kuairand_collate_fn,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    if bool(config.get("use_user_group_sampler", False)):
        train_batch_sampler = UserGroupBatchSampler(
            user_ids=train_ds.arrays["user_id"],
            labels=train_ds.arrays["label"],
            batch_size=batch_size,
            samples_per_user=int(config.get("user_group_samples_per_user", 4)),
            label_threshold=float(config.get("pairwise_label_threshold", 0.5)),
            seed=int(config.get("seed", 2025)),
            drop_last=bool(config.get("user_group_drop_last", False)),
        )
        logger.info(
            "Using user-group batch sampler | batches=%d | effective_batch_size=%d | users_per_batch=%d | samples_per_user=%d",
            len(train_batch_sampler),
            train_batch_sampler.batch_size,
            train_batch_sampler.users_per_batch,
            train_batch_sampler.samples_per_user,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            **common_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            **common_loader_kwargs,
        )
    aux_rank_loader = None
    if bool(config.get("use_aux_rank_loader", False)):
        aux_rank_batch_sampler = UserGroupBatchSampler(
            user_ids=train_ds.arrays["user_id"],
            labels=train_ds.arrays["label"],
            batch_size=int(config.get("aux_rank_batch_size", batch_size)),
            samples_per_user=int(config.get("aux_rank_samples_per_user", 4)),
            label_threshold=float(config.get("pairwise_label_threshold", 0.5)),
            seed=int(config.get("seed", 2025)) + 1009,
            drop_last=bool(config.get("aux_rank_drop_last", False)),
        )
        aux_rank_num_workers = int(config.get("aux_rank_num_workers", 0))
        aux_rank_loader_kwargs = {
            "num_workers": aux_rank_num_workers,
            "collate_fn": kuairand_collate_fn,
            "pin_memory": pin_memory,
        }
        if aux_rank_num_workers > 0:
            aux_rank_loader_kwargs["persistent_workers"] = bool(config.get("aux_rank_persistent_workers", False))
            aux_rank_loader_kwargs["prefetch_factor"] = int(config.get("aux_rank_prefetch_factor", 2))
        aux_rank_loader = DataLoader(
            train_ds,
            batch_sampler=aux_rank_batch_sampler,
            **aux_rank_loader_kwargs,
        )
        logger.info(
            "Using auxiliary rank loader | batches=%d | effective_batch_size=%d | users_per_batch=%d | samples_per_user=%d | every_n_steps=%d | loss_weight=%.6f",
            len(aux_rank_batch_sampler),
            aux_rank_batch_sampler.batch_size,
            aux_rank_batch_sampler.users_per_batch,
            aux_rank_batch_sampler.samples_per_user,
            int(config.get("aux_rank_every_n_steps", 10)),
            float(config.get("aux_rank_loss_weight", 0.01)),
        )
    random_aux_loader = None
    if bool(config.get("use_random_aux", False)):
        default_random_aux_path = data_dir / ("random_aux_train_debug.parquet" if args.debug else "random_aux_train.parquet")
        random_aux_path = Path(str(config.get("random_aux_data_path", default_random_aux_path)))
        if not random_aux_path.is_absolute():
            random_aux_path = PROJECT_ROOT / random_aux_path
        if args.debug and random_aux_path.name == "random_aux_train.parquet":
            debug_random_aux_path = random_aux_path.with_name("random_aux_train_debug.parquet")
            if debug_random_aux_path.exists() or debug_random_aux_path.with_suffix(".pkl").exists():
                random_aux_path = debug_random_aux_path
        logger.info("Loading random auxiliary dataset: %s", random_aux_path)
        random_aux_ds = KuaiRandDataset(random_aux_path)
        random_aux_num_workers = 0 if args.debug else int(config.get("random_aux_num_workers", 0))
        random_aux_loader_kwargs = {
            "num_workers": random_aux_num_workers,
            "collate_fn": kuairand_collate_fn,
            "pin_memory": pin_memory,
        }
        if random_aux_num_workers > 0:
            random_aux_loader_kwargs["persistent_workers"] = bool(config.get("random_aux_persistent_workers", False))
            random_aux_loader_kwargs["prefetch_factor"] = int(config.get("random_aux_prefetch_factor", 2))
        random_aux_loader = DataLoader(
            random_aux_ds,
            batch_size=int(config.get("random_aux_batch_size", batch_size)),
            shuffle=True,
            drop_last=bool(config.get("random_aux_drop_last", True)),
            **random_aux_loader_kwargs,
        )
        logger.info(
            "Using random auxiliary loader | rows=%d | batches=%d | every_n_steps=%d | loss_weight=%.6f",
            len(random_aux_ds),
            len(random_aux_loader),
            int(config.get("random_aux_every_n_steps", 8)),
            float(config.get("random_loss_weight", 0.05)),
        )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    model = build_model(config, feature_maps)
    n_params = count_parameters(model)
    logger.info("Model parameter count | total=%d | trainable=%d", n_params["total"], n_params["trainable"])

    requested_device = config.get("device", "cuda")
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    first_batch = next(iter(train_loader))
    log_batch_shape_summary(logger, first_batch)
    if bool(config.get("use_video_semantic_emb", False)):
        logger.info(
            "Semantic batch diagnostics | target_semantic_emb shape=%s | hist_semantic_emb shape=%s | simtier_features shape=%s | semantic_missing_rate=%.6f | simtier_nan_count=%d",
            tuple(first_batch["target_semantic_emb"].shape) if "target_semantic_emb" in first_batch else None,
            tuple(first_batch["hist_semantic_emb"].shape) if "hist_semantic_emb" in first_batch else None,
            tuple(first_batch["simtier_features"].shape) if "simtier_features" in first_batch else None,
            float(first_batch.get("semantic_missing_flag", torch.zeros(1)).float().mean().item()),
            int(torch.isnan(first_batch.get("simtier_features", torch.zeros(1))).sum().item()),
        )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-6)),
    )

    trainer = CTRTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        logger=logger,
    )

    ckpt_base = experiment_name if experiment_name else f"{config['model_name']}_{mode_name}{run_suffix}"
    if args.debug:
        ckpt_base = f"{ckpt_base}_debug"
    ckpt_path = ckpt_dir / f"{ckpt_base}_best.pt"
    best_ckpt_path = trainer.fit(train_loader, valid_loader, ckpt_path, aux_rank_loader=aux_rank_loader, random_aux_loader=random_aux_loader)

    state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    logger.info("Loading test dataset: %s", test_file)
    test_ds = KuaiRandDataset(test_file, config=config, feature_maps=feature_maps)
    logger.info("Loaded test dataset size: %d", len(test_ds))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        "Test metrics | auc=%.6f | gauc=%.6f | logloss=%.6f | extra_test_metrics=%s",
        test_metrics["auc"],
        test_metrics["gauc"],
        test_metrics["logloss"],
        {k: round(float(v), 6) for k, v in test_metrics.items() if k not in {"auc", "gauc", "logloss"}},
    )

    result = {
        "model_name": config["model_name"],
        "experiment_name": experiment_name,
        "run_name": run_name,
        "mode": mode_name,
        "best_checkpoint": str(best_ckpt_path),
        "test_metrics": test_metrics,
    }
    if bool(config.get("use_video_semantic_emb", False)):
        diagnostics = {
            "semantic_missing_rate": float(first_batch.get("semantic_missing_flag", torch.zeros(1)).float().mean().item()),
            "simtier_nan_count": int(torch.isnan(first_batch.get("simtier_features", torch.zeros(1))).sum().item()),
            "simtier_feature_dim": int(feature_maps.get("simtier_dim", 0)),
            "semantic_gate_mean": float(test_metrics.get("semantic_gate_mean", 0.0)),
            "semantic_gate_std": float(test_metrics.get("semantic_gate_std", 0.0)),
            "short_history_non_empty_ratio": float(test_metrics.get("short_history_non_empty_mean", 0.0)),
            "long_history_non_empty_ratio": float(test_metrics.get("long_history_non_empty_mean", 0.0)),
            "short_sem_interest_norm_mean": float(test_metrics.get("short_sem_interest_norm_mean", 0.0)),
            "long_sem_interest_norm_mean": float(test_metrics.get("long_sem_interest_norm_mean", 0.0)),
            "target_semantic_repr_norm_mean": float(test_metrics.get("target_semantic_repr_norm_mean", 0.0)),
        }
        save_json(diagnostics, out_dir / f"{experiment_name}_diagnostics.json")
    output_run_name = f"{run_name}_debug" if args.debug and run_name else run_name
    if not run_name:
        save_json(result, out_dir / "test_metrics.json")
        save_json(result, out_dir / f"test_metrics_{config['model_name']}.json")
    else:
        save_json(result, out_dir / f"{output_run_name}_test_metrics.json")
        save_json(result, out_dir / f"test_metrics_{config['model_name']}_{output_run_name}.json")


if __name__ == "__main__":
    main()
