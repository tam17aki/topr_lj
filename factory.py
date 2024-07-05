# -*- coding: utf-8 -*-
"""A Python module which provides optimizer, scheduler, and customized loss.

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

import math

import torch
from timm.scheduler import CosineLRScheduler
from torch import nn, optim
from torch.optim.optimizer import Optimizer

import config
from model import TOPRNet


def get_optimizer(model: TOPRNet) -> Optimizer:
    """Instantiate optimizer.

    Args:
        model (nn.Module): network parameters.

    Returns:
        optimizer (Optimizer): RAdam or AdamW.
    """
    cfg = config.OptimizerConfig()
    if cfg.name == "RAdam":
        return optim.RAdam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            decoupled_weight_decay=cfg.decoupled_weight_decay,
        )
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def get_lr_scheduler(optimizer: Optimizer) -> CosineLRScheduler:
    """Instantiate scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

    Returns:
        lr_scheduler (CosineLRScheduler): cosine scheduler with warmup.
    """
    cfg = config.SchedulerConfig()
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.t_initial,
        lr_min=cfg.lr_min,
        warmup_t=cfg.warmup_t,
        warmup_lr_init=cfg.warmup_lr_init,
        warmup_prefix=True,
    )
    return lr_scheduler


class CustomLoss(nn.Module):
    """Custom loss."""

    def __init__(self, model: TOPRNet) -> None:
        """Initialize class.

        Args:
            model (TOPRNet): neural network to estimate phase differences.
        """
        super().__init__()
        self.feat_cfg = config.FeatureConfig()
        self.train_cfg = config.TrainingConfig()
        self.model = model

    def _tpd2bpd(self, tpd: torch.Tensor) -> torch.Tensor:
        """Modify TPD to BPD.

        Args:
            tpd (Tensor): oracle backward TPD. [B, T-1, K]

        Returns:
            bpd (Tensor): oracle backward BPD. [B, T-1, K]
        """
        win_len = self.feat_cfg.win_length
        hop_len = self.feat_cfg.hop_length
        n_batch, n_frame, _ = tpd.shape
        pi_tensor = torch.Tensor([math.pi]).cuda()
        k = torch.arange(0, win_len // 2 + 1).cuda()
        angle_freq = (2 * pi_tensor / win_len) * k * hop_len
        angle_freq = angle_freq.unsqueeze(0).expand(n_frame, len(k))
        angle_freq = angle_freq.unsqueeze(0).expand(n_batch, n_frame, len(k))
        bpd = torch.angle(torch.exp(1j * (tpd - angle_freq)))
        return bpd

    def _compute_bpd_loss(
        self, predicted: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss of backward baseband phase difference (BPD).

        Args:
            predicted (Tensor): estimated backward BPD. [B, T-1, K]
            reference (Tensor): ground-truth phase spectrum. [B, T-1, K]

        Returns:
            loss (Tensor): cosine loss of backward BPD.
        """
        oracle_tpd = reference[:, 1:, :] - reference[:, :-1, :]
        oracle_bpd = self._tpd2bpd(oracle_tpd)
        diff = predicted[:, :-1, :] - oracle_bpd
        loss = torch.sum(-torch.cos(diff), dim=-1)  # sum along frequency axis
        loss = torch.sum(loss, dim=-1)  # sum along time axis
        return loss.mean()  # average along batch axis

    def _compute_fpd_loss(
        self, predicted: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss of backward phase difference for frequency (FPD).

        Args:
            predicted (Tensor): estimated backward FPD. [B, T, K]
            reference (Tensor): ground-truth phase spectrum. [B, T, K]

        Returns:
            loss (Tensor): cosine loss of backward FPD.
        """
        oracle_fpd = reference[:, :, 1:] - reference[:, :, :-1]
        diff = predicted[:, :, :-1] - oracle_fpd
        loss = torch.sum(-torch.cos(diff), dim=-1)  # sum along frequency axis
        loss = torch.sum(loss, dim=-1)  # sum along time axis
        return loss.mean()  # average along batch axis

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor], mode: str
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            batch (Tuple): tuple of minibatch.
            mode (str): specify loss function; 'bpd' or 'fpd'.

        Returns:
            loss (Tensor): cosine loss of BPD or FPD.
        """
        logmag_batch, phase_batch = batch
        logmag_batch = logmag_batch.cuda().float()  # [B*T, L+1, K]
        phase_batch = phase_batch.cuda().float()  # [B, T, K]
        predicted = self.model.forward(logmag_batch)  # [B*T, 1, K]
        predicted = predicted.squeeze()  # [B*T, K]
        predicted = predicted.reshape(
            self.train_cfg.n_batch, -1, self.feat_cfg.n_fft // 2 + 1
        )  # [B, T, K]
        if mode == "bpd":
            loss = self._compute_bpd_loss(predicted, phase_batch)
        else:
            loss = self._compute_fpd_loss(predicted, phase_batch)
        return loss
