# -*- coding: utf-8 -*-
"""Dataset definition for Two-stage Online/Offline Phase Reconstruction (TOPR).

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from config import ModelConfig, PathConfig, TrainingConfig


@dataclass
class FeatPath:
    """Paths for features."""

    logmag: list[str]
    phase: list[str]


class TOPRDataset(Dataset[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]):
    """Dataset for TOPR."""

    def __init__(self, feat_paths: FeatPath) -> None:
        """Initialize class."""
        self.logmag_paths = feat_paths.logmag
        self.phase_paths = feat_paths.phase

    def __getitem__(
        self, idx: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Get a pair of input and target.

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return (np.load(self.logmag_paths[idx]), np.load(self.phase_paths[idx]))

    def __len__(self) -> int:
        """Return the size of the dataset.

        Returns:
            int: size of the dataset
        """
        return len(self.logmag_paths)


def collate_fn_topr(
    batch: list[npt.NDArray[np.float32]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for TOPR.

    Args:
        batch (Tuple): tuple of minibatch.
        cfg (DictConfig): configuration in YAML format.

    Returns:
        tuple: a batch of inputs and targets.
    """
    cfg = ModelConfig()
    batch_temp = [x[0] for x in batch]
    logmag_feats = torch.tensor(np.array(batch_temp).astype(np.float32))
    logmag_feats = logmag_feats.unfold(1, cfg.n_lookback + cfg.n_lookahead + 1, 1)
    _, _, n_fbin, width = logmag_feats.shape
    logmag_feats = logmag_feats.reshape(-1, n_fbin, width)
    logmag_feats = logmag_feats.transpose(2, 1)

    batch_temp = [x[1] for x in batch]
    phase_feats = torch.tensor(np.array(batch_temp).astype(np.float32))
    _, n_frame, _ = phase_feats.shape
    phase_feats = phase_feats[:, cfg.n_lookback : n_frame - cfg.n_lookahead, :]
    return (logmag_feats, phase_feats)


def get_dataloader() -> (
    DataLoader[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
):
    """Get data loaders for training and validation.

    Args:
        cfg (DictConfig): configuration in YAML format.

    Returns:
        dict: Data loaders.
    """
    train_cfg = TrainingConfig()
    path_cfg = PathConfig()
    wav_list = os.listdir(
        os.path.join(path_cfg.root_dir, path_cfg.data_dir, path_cfg.split_dir)
    )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()

    feat_dir = os.path.join(path_cfg.root_dir, path_cfg.feat_dir, "train")
    feat_paths = FeatPath(
        logmag=[
            os.path.join(feat_dir, f"{utt_id}-feats_logmag.npy") for utt_id in utt_list
        ],
        phase=[
            os.path.join(feat_dir, f"{utt_id}-feats_phase.npy") for utt_id in utt_list
        ],
    )

    data_loaders = DataLoader(
        TOPRDataset(feat_paths),
        batch_size=train_cfg.n_batch,
        collate_fn=collate_fn_topr,
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )
    return data_loaders
