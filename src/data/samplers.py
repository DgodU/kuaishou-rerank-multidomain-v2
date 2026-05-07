from __future__ import annotations

import math
from typing import Iterator, List

import numpy as np
from torch.utils.data import Sampler


class UserGroupBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        user_ids: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        samples_per_user: int = 4,
        label_threshold: float = 0.5,
        seed: int = 2025,
        drop_last: bool = False,
    ):
        self.user_ids = np.asarray(user_ids)
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.samples_per_user = max(2, int(samples_per_user))
        self.label_threshold = float(label_threshold)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.users_per_batch = max(1, self.batch_size // self.samples_per_user)
        self.batch_size = self.users_per_batch * self.samples_per_user
        self.epoch = 0

        self.user_pos: dict[int, np.ndarray] = {}
        self.user_neg: dict[int, np.ndarray] = {}
        self.user_all: dict[int, np.ndarray] = {}
        valid_users = []

        order = np.argsort(self.user_ids, kind="stable")
        sorted_users = self.user_ids[order]
        boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
        for group in np.split(order, boundaries):
            uid = self.user_ids[group[0]]
            idx = group
            user_labels = self.labels[idx]
            pos_idx = idx[user_labels > self.label_threshold]
            neg_idx = idx[user_labels <= self.label_threshold]
            if pos_idx.size > 0 and neg_idx.size > 0:
                uid_int = int(uid)
                self.user_pos[uid_int] = pos_idx
                self.user_neg[uid_int] = neg_idx
                self.user_all[uid_int] = idx
                valid_users.append(uid_int)

        if not valid_users:
            raise ValueError("UserGroupBatchSampler requires at least one user with both positive and negative samples")

        self.valid_users = np.asarray(valid_users, dtype=np.int64)
        if self.drop_last:
            self.num_batches = len(self.user_ids) // self.batch_size
        else:
            self.num_batches = math.ceil(len(self.user_ids) / self.batch_size)
        self.num_batches = max(1, int(self.num_batches))

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        replace_users = self.valid_users.size < self.users_per_batch

        for _ in range(self.num_batches):
            batch_indices = []
            batch_users = rng.choice(self.valid_users, size=self.users_per_batch, replace=replace_users)
            for uid_np in batch_users:
                uid = int(uid_np)
                pos_idx = self.user_pos[uid]
                neg_idx = self.user_neg[uid]
                all_idx = self.user_all[uid]

                selected = [
                    int(rng.choice(pos_idx)),
                    int(rng.choice(neg_idx)),
                ]
                if self.samples_per_user > 2:
                    fill = rng.choice(all_idx, size=self.samples_per_user - 2, replace=all_idx.size < self.samples_per_user - 2)
                    selected.extend(int(x) for x in fill)
                rng.shuffle(selected)
                batch_indices.extend(selected)

            rng.shuffle(batch_indices)
            yield batch_indices
