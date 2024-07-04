# -*- coding: utf-8 -*-
"""Training script for Two-stage Online/Offline Phase Reconstruction (TOPR).

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
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from progressbar import progressbar as prg
from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary

from config import (
    FeatureConfig,
    ModelConfig,
    OptimizerConfig,
    PathConfig,
    PreProcessConfig,
    SchedulerConfig,
    TrainingConfig,
)
from dataset import get_dataloader
from factory import (
    CustomLoss,
    get_lr_scheduler,
    get_optimizer,
)
from model import TOPRNet


class TrainingModules(NamedTuple):
    """Training Modules."""

    dataloader: DataLoader[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
    model: TOPRNet
    loss_func: CustomLoss
    optimizer: Optimizer
    lr_scheduler: CosineLRScheduler | None


def print_config() -> None:
    """Print all configurations for training."""
    for cfg in (
        PathConfig(),
        PreProcessConfig(),
        FeatureConfig(),
        ModelConfig(),
        OptimizerConfig(),
        SchedulerConfig(),
        TrainingConfig(),
    ):
        print(cfg)


def get_training_modules() -> TrainingModules:
    """Instantiate modules for training.

    Args:
       None.

    Returns:
       modules (TrainingModules): modules required for training.
    """
    cfg = TrainingConfig()
    dataloader = get_dataloader()
    model = TOPRNet().cuda()
    loss_func = CustomLoss(model)
    optimizer = get_optimizer(model)
    lr_scheduler = None
    if cfg.use_scheduler:
        lr_scheduler = get_lr_scheduler(optimizer)
    modules = TrainingModules(dataloader, model, loss_func, optimizer, lr_scheduler)
    summary(model)
    return modules


def training_loop(modules: TrainingModules, mode: str) -> None:
    """Perform training loop.

    Args:
        modules (TrainingModules): modules required for training.
        mode (str): string to specfiy training mode (BPD or FPD).

    Returns:
       None.
    """
    cfg = TrainingConfig()
    dataloader, model, loss_func, optimizer, lr_scheduler = modules
    model.train()
    n_epoch = cfg.n_epoch + 1
    for epoch in prg(
        range(1, n_epoch), prefix="Model training: ", suffix=" ", redirect_stdout=False
    ):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_func.forward(batch, mode)
            epoch_loss += loss.item()
            loss.backward()
            if cfg.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        epoch_loss = epoch_loss / len(dataloader)
        if epoch == 1 or epoch % cfg.report_interval == 0:
            print(f"\nEpoch {epoch}: loss = {epoch_loss:.12f} ")


def save_checkpoint(modules: TrainingModules, mode: str) -> None:
    """Save checkpoint.

    Args:
        modules (TrainingModules): modules required for training.
        mode (str): string to specfiy training mode (BPD or FPD).

    Returns:
       None.
    """
    path_cfg = PathConfig()
    model = modules.model
    model_dir = os.path.join(path_cfg.root_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_file = ""
    if mode == "bpd":
        model_file = os.path.join(model_dir, path_cfg.model_file + ".bpd.pth")
    elif mode == "fpd":
        model_file = os.path.join(model_dir, path_cfg.model_file + ".fpd.pth")
    torch.save(model.state_dict(), f=model_file)


def main() -> None:
    """Perform model training."""
    modules = get_training_modules()
    training_loop(modules, "bpd")
    save_checkpoint(modules, "bpd")

    modules = get_training_modules()
    training_loop(modules, "fpd")
    save_checkpoint(modules, "fpd")


if __name__ == "__main__":
    print_config()
    main()
