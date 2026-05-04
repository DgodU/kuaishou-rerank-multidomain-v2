from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.metrics import compute_all_metrics


class CTRTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        config: Dict,
        logger,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger
        self.use_ema = bool(config.get("use_ema", False))
        self.ema_decay = float(config.get("ema_decay", 0.999))
        self.ema_warmup_steps = int(config.get("ema_warmup_steps", 0))
        self.global_step = 0
        self.ema_state_dict = None

    def _clone_model_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _update_ema(self) -> None:
        if not self.use_ema:
            return

        model_state = self.model.state_dict()
        if self.ema_state_dict is None:
            self.ema_state_dict = {k: v.detach().clone() for k, v in model_state.items()}
            return

        if self.global_step <= self.ema_warmup_steps:
            for k, v in model_state.items():
                self.ema_state_dict[k].copy_(v.detach())
            return

        for k, v in model_state.items():
            if torch.is_floating_point(v):
                self.ema_state_dict[k].mul_(self.ema_decay).add_(v.detach(), alpha=1.0 - self.ema_decay)
            else:
                self.ema_state_dict[k].copy_(v.detach())

    def _load_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _select_main_train_logit(self, out: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]) -> torch.Tensor:
        if bool(self.config.get("use_position_bias_tower", False)):
            train_logit = str(self.config.get("pal_train_logit", "observed")).lower()
            if train_logit == "relevance" and "relevance_logit" in out:
                return out["relevance_logit"]
            if "observed_logit" in out:
                return out["observed_logit"]
        if bool(self.config.get("use_rank_calib_split", False)) and "final_logit" in out:
            return out["final_logit"]
        return out["logit"]

    def _best_metric_name(self) -> str:
        if bool(self.config.get("use_position_bias_tower", False)):
            return "relevance_gauc"
        if bool(self.config.get("use_rank_calib_split", False)):
            return "ranking_gauc"
        return "gauc"

    def _compute_train_loss(
        self,
        out: Dict[str, torch.Tensor | Dict[str, torch.Tensor]],
        label: torch.Tensor,
        user_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        main_logit = self._select_main_train_logit(out)
        main_loss = self.criterion(main_logit, label)

        pairwise_loss = torch.zeros((), device=label.device)
        if bool(self.config.get("use_pairwise_loss", False)):
            pairwise_loss = self._compute_pairwise_loss(
                logits=main_logit,
                labels=label,
                user_ids=user_id,
            )

        aux_loss = torch.zeros((), device=label.device)
        if bool(self.config.get("use_mbc_aux_loss", False)) and "branch_logits" in out:
            branch_logits = out.get("branch_logits", {}) or {}
            branch_losses = []
            for _, b_logit in branch_logits.items():
                branch_losses.append(self.criterion(b_logit, label))
            if branch_losses:
                aux_loss = torch.stack(branch_losses).mean()

        diversity_loss = torch.zeros((), device=label.device)
        if bool(self.config.get("use_mbc_diversity_loss", False)) and "branch_vectors" in out:
            branch_vectors = out.get("branch_vectors", {}) or {}
            z_efgc = branch_vectors.get("efgc")
            z_cross = branch_vectors.get("cross")
            z_deep = branch_vectors.get("deep")
            if z_efgc is not None and z_cross is not None and z_deep is not None:
                n_efgc = torch.nn.functional.normalize(z_efgc, dim=-1)
                n_cross = torch.nn.functional.normalize(z_cross, dim=-1)
                n_deep = torch.nn.functional.normalize(z_deep, dim=-1)

                cos_ec = (n_efgc * n_cross).sum(dim=-1).mean()
                cos_ed = (n_efgc * n_deep).sum(dim=-1).mean()
                cos_cd = (n_cross * n_deep).sum(dim=-1).mean()
                diversity_loss = (cos_ec.pow(2) + cos_ed.pow(2) + cos_cd.pow(2)) / 3.0

        total_loss = main_loss
        if bool(self.config.get("use_mbc_aux_loss", False)):
            total_loss = total_loss + float(self.config.get("mbc_aux_loss_weight", 0.1)) * aux_loss
        if bool(self.config.get("use_mbc_diversity_loss", False)):
            total_loss = total_loss + float(self.config.get("mbc_diversity_loss_weight", 1e-4)) * diversity_loss
        if bool(self.config.get("use_pairwise_loss", False)):
            total_loss = total_loss + float(self.config.get("pairwise_loss_weight", 0.1)) * pairwise_loss

        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "pairwise_loss": pairwise_loss,
            "aux_loss": aux_loss,
            "diversity_loss": diversity_loss,
        }

    def _compute_pairwise_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        label_thr = float(self.config.get("pairwise_label_threshold", 0.5))
        pos_mask = labels > label_thr
        neg_mask = ~pos_mask

        if not pos_mask.any() or not neg_mask.any():
            return torch.zeros((), device=labels.device)

        _, inverse = torch.unique(user_ids, return_inverse=True)
        n_users = int(inverse.max().item()) + 1

        inf = torch.tensor(float("inf"), device=logits.device, dtype=logits.dtype)
        neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)

        pos_logits = torch.where(pos_mask, logits, inf)
        neg_logits = torch.where(neg_mask, logits, neg_inf)

        hard_pos = torch.full((n_users,), float("inf"), device=logits.device, dtype=logits.dtype)
        hard_neg = torch.full((n_users,), float("-inf"), device=logits.device, dtype=logits.dtype)
        hard_pos.scatter_reduce_(0, inverse, pos_logits, reduce="amin", include_self=True)
        hard_neg.scatter_reduce_(0, inverse, neg_logits, reduce="amax", include_self=True)

        one = torch.ones_like(inverse)
        pos_count = torch.zeros(n_users, device=inverse.device, dtype=inverse.dtype)
        neg_count = torch.zeros(n_users, device=inverse.device, dtype=inverse.dtype)
        pos_count.scatter_add_(0, inverse, torch.where(pos_mask, one, torch.zeros_like(one)))
        neg_count.scatter_add_(0, inverse, torch.where(neg_mask, one, torch.zeros_like(one)))

        valid = (pos_count > 0) & (neg_count > 0)
        if not valid.any():
            return torch.zeros((), device=labels.device)

        pairwise_mode = str(self.config.get("pairwise_loss_mode", "hardest")).lower()
        if pairwise_mode != "all_pairs":
            diff_tensor = hard_pos[valid] - hard_neg[valid]
            return torch.nn.functional.softplus(-diff_tensor).mean()

        max_pairs_per_user = int(self.config.get("pairwise_max_pairs_per_user", 256))
        same_user = inverse.unsqueeze(1).eq(inverse.unsqueeze(0))
        pair_mask = same_user & pos_mask.unsqueeze(1) & neg_mask.unsqueeze(0)
        diffs = logits.unsqueeze(1) - logits.unsqueeze(0)
        diffs = diffs[pair_mask]

        if diffs.numel() == 0:
            return torch.zeros((), device=labels.device)

        max_pairs = max_pairs_per_user * int(valid.sum().item())
        if max_pairs > 0 and diffs.numel() > max_pairs:
            pair_idx = torch.randperm(diffs.numel(), device=diffs.device)[:max_pairs]
            diffs = diffs[pair_idx]
        return torch.nn.functional.softplus(-diffs).mean()

    def train_one_epoch(
        self,
        loader: DataLoader,
        aux_rank_loader: DataLoader | None = None,
        random_aux_loader: DataLoader | None = None,
    ) -> Dict[str, float]:
        self.model.train()
        total_losses = []
        main_losses = []
        pairwise_losses = []
        aux_losses = []
        diversity_losses = []
        aux_rank_losses = []
        random_aux_losses = []
        random_aux_steps = 0
        aux_rank_iter = iter(aux_rank_loader) if aux_rank_loader is not None else None
        random_aux_iter = iter(random_aux_loader) if random_aux_loader is not None else None
        aux_rank_every_n_steps = max(1, int(self.config.get("aux_rank_every_n_steps", 10)))
        aux_rank_loss_weight = float(self.config.get("aux_rank_loss_weight", 0.01))
        random_aux_every_n_steps = max(1, int(self.config.get("random_aux_every_n_steps", 8)))
        random_loss_weight = float(self.config.get("random_loss_weight", 0.05))

        for step_idx, batch in enumerate(loader, start=1):
            batch = self._move_batch(batch)
            out = self.model(batch)
            loss_dict = self._compute_train_loss(out, batch["label"], batch["user_id"])  # logit/label: [B]
            loss = loss_dict["total_loss"]

            if random_aux_iter is not None and step_idx % random_aux_every_n_steps == 0:
                try:
                    random_batch = next(random_aux_iter)
                except StopIteration:
                    random_aux_iter = iter(random_aux_loader)
                    random_batch = next(random_aux_iter)
                random_batch = self._move_batch(random_batch)
                random_out = self.model(random_batch)
                random_logit = random_out.get("random_logit", random_out["logit"])
                random_loss = self.criterion(random_logit, random_batch["label"])
                loss = loss + random_loss_weight * random_loss
                random_aux_losses.append(random_loss.detach().item())
                random_aux_steps += 1
                if bool(self.config.get("_debug_shapes", False)) and random_aux_steps == 1:
                    self.logger.info(
                        "Random auxiliary debug | random_batch_size=%d | random_logit_shape=%s | random_aux_loss=%.6f",
                        int(random_batch["label"].shape[0]),
                        tuple(random_logit.shape),
                        float(random_loss.detach().item()),
                    )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            self._update_ema()

            total_losses.append(loss.detach().item())
            main_losses.append(loss_dict["main_loss"].detach().item())
            pairwise_losses.append(loss_dict["pairwise_loss"].detach().item())
            aux_losses.append(loss_dict["aux_loss"].detach().item())
            diversity_losses.append(loss_dict["diversity_loss"].detach().item())

            if aux_rank_iter is not None and step_idx % aux_rank_every_n_steps == 0:
                try:
                    rank_batch = next(aux_rank_iter)
                except StopIteration:
                    aux_rank_iter = iter(aux_rank_loader)
                    rank_batch = next(aux_rank_iter)

                rank_batch = self._move_batch(rank_batch)
                rank_out = self.model(rank_batch)
                aux_rank_loss = self._compute_pairwise_loss(
                    logits=rank_out["logit"],
                    labels=rank_batch["label"],
                    user_ids=rank_batch["user_id"],
                )
                rank_loss = aux_rank_loss_weight * aux_rank_loss

                self.optimizer.zero_grad(set_to_none=True)
                rank_loss.backward()
                self.optimizer.step()
                self.global_step += 1
                self._update_ema()
                aux_rank_losses.append(aux_rank_loss.detach().item())

        if not total_losses:
            return {
                "train_total_loss": 0.0,
                "train_main_loss": 0.0,
                "train_pairwise_loss": 0.0,
                "train_aux_rank_loss": 0.0,
                "train_aux_loss": 0.0,
                "train_diversity_loss": 0.0,
                "train_random_aux_loss": 0.0,
                "random_aux_steps_per_epoch": 0.0,
                "random_aux_batch_count": float(len(random_aux_loader)) if random_aux_loader is not None else 0.0,
            }
        return {
            "train_total_loss": float(np.mean(total_losses)),
            "train_main_loss": float(np.mean(main_losses)),
            "train_pairwise_loss": float(np.mean(pairwise_losses)),
            "train_aux_rank_loss": float(np.mean(aux_rank_losses)) if aux_rank_losses else 0.0,
            "train_aux_loss": float(np.mean(aux_losses)),
            "train_diversity_loss": float(np.mean(diversity_losses)),
            "train_random_aux_loss": float(np.mean(random_aux_losses)) if random_aux_losses else 0.0,
            "random_aux_steps_per_epoch": float(random_aux_steps),
            "random_aux_batch_count": float(len(random_aux_loader)) if random_aux_loader is not None else 0.0,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        y_true_all = []
        pred_all: Dict[str, list[np.ndarray]] = {"": []}
        user_ids_all = []

        for batch in loader:
            batch = self._move_batch(batch)
            out = self.model(batch)

            y_true_all.append(batch["label"].detach().cpu().numpy())
            pred_all[""].append(out["pred"].detach().cpu().numpy())
            for key, prefix in [
                ("relevance_logit", "relevance"),
                ("observed_logit", "observed"),
                ("ranking_logit", "ranking"),
                ("final_logit", "final"),
            ]:
                if key in out:
                    pred_all.setdefault(prefix, []).append(torch.sigmoid(out[key]).detach().cpu().numpy())
            user_ids_all.append(batch["user_id"].detach().cpu().numpy())

        if not y_true_all:
            return {"auc": 0.5, "gauc": 0.5, "logloss": 0.0}

        y_true = np.concatenate(y_true_all)
        user_ids = np.concatenate(user_ids_all)
        metrics = {}
        for prefix, preds in pred_all.items():
            if not preds:
                continue
            m = compute_all_metrics(y_true, np.concatenate(preds), user_ids)
            if prefix == "":
                metrics.update(m)
            else:
                metrics[f"{prefix}_auc"] = m["auc"]
                metrics[f"{prefix}_gauc"] = m["gauc"]
                metrics[f"{prefix}_logloss"] = m["logloss"]
        return metrics

    @torch.no_grad()
    def evaluate_ema(self, loader: DataLoader) -> Dict[str, float]:
        if not self.use_ema or self.ema_state_dict is None:
            return self.evaluate(loader)

        model_state = self._clone_model_state()
        self._load_state(self.ema_state_dict)
        metrics = self.evaluate(loader)
        self._load_state(model_state)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        checkpoint_path: str | Path,
        aux_rank_loader: DataLoader | None = None,
        random_aux_loader: DataLoader | None = None,
    ) -> str:
        best_gauc = -1.0
        best_epoch = -1
        best_auc = -1.0
        best_auc_epoch = -1
        wait = 0

        epochs = int(self.config.get("epochs", 20))
        patience = int(self.config.get("early_stop_patience", 3))
        save_best_auc_checkpoint = bool(self.config.get("save_best_auc_checkpoint", False))

        checkpoint_path = str(checkpoint_path)
        checkpoint_auc_path = checkpoint_path.replace(".pt", "_best_auc.pt")

        for epoch in range(1, epochs + 1):
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            aux_batch_sampler = getattr(aux_rank_loader, "batch_sampler", None) if aux_rank_loader is not None else None
            if hasattr(aux_batch_sampler, "set_epoch"):
                aux_batch_sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(train_loader, aux_rank_loader=aux_rank_loader, random_aux_loader=random_aux_loader)
            valid_metrics = self.evaluate_ema(valid_loader) if self.use_ema else self.evaluate(valid_loader)
            best_metric_name = self._best_metric_name()
            current_best_metric = float(valid_metrics.get(best_metric_name, valid_metrics["gauc"]))

            self.logger.info(
                "Epoch %d | train_total_loss=%.6f | train_main_loss=%.6f | train_pairwise_loss=%.6f | train_aux_rank_loss=%.6f | train_random_aux_loss=%.6f | random_aux_steps_per_epoch=%.0f | train_aux_loss=%.6f | train_diversity_loss=%.6f | valid_auc=%.6f | valid_gauc=%.6f | valid_logloss=%.6f | best_metric_name=%s | best_metric=%.6f | extra_valid_metrics=%s",
                epoch,
                train_stats["train_total_loss"],
                train_stats["train_main_loss"],
                train_stats["train_pairwise_loss"],
                train_stats["train_aux_rank_loss"],
                train_stats["train_random_aux_loss"],
                train_stats["random_aux_steps_per_epoch"],
                train_stats["train_aux_loss"],
                train_stats["train_diversity_loss"],
                valid_metrics["auc"],
                valid_metrics["gauc"],
                valid_metrics["logloss"],
                best_metric_name,
                current_best_metric,
                {k: round(float(v), 6) for k, v in valid_metrics.items() if k not in {"auc", "gauc", "logloss"}},
            )

            if current_best_metric > best_gauc:
                best_gauc = current_best_metric
                best_epoch = epoch
                wait = 0
                torch.save(
                    {
                        "model_state_dict": self.ema_state_dict if self.use_ema and self.ema_state_dict is not None else self.model.state_dict(),
                        "config": self.config,
                        "best_epoch": best_epoch,
                        "best_valid_gauc": best_gauc,
                        "use_ema": self.use_ema,
                    },
                    checkpoint_path,
                )
                self.logger.info("Saved new best checkpoint to %s", checkpoint_path)
            else:
                wait += 1
                if wait >= patience:
                    self.logger.info(
                        "Early stopping at epoch %d (best_epoch=%d, best_gauc=%.6f)",
                        epoch,
                        best_epoch,
                        best_gauc,
                    )
                    break

            if save_best_auc_checkpoint and valid_metrics["auc"] > best_auc:
                best_auc = valid_metrics["auc"]
                best_auc_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": self.ema_state_dict if self.use_ema and self.ema_state_dict is not None else self.model.state_dict(),
                        "config": self.config,
                        "best_epoch": best_auc_epoch,
                        "best_valid_auc": best_auc,
                        "use_ema": self.use_ema,
                    },
                    checkpoint_auc_path,
                )
                self.logger.info("Saved best-AUC checkpoint to %s", checkpoint_auc_path)

        return checkpoint_path
