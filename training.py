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

import torch
from torch import nn
from torchinfo import summary
from tqdm import tqdm

import config
from dataset import get_dataloader
from factory import CustomLoss, get_lr_scheduler, get_optimizer
from model import TOPRNet


def training_loop(mode: str) -> TOPRNet:
    """Perform training loop.

    Args:
        mode (str): string to specfiy training mode (BPD or FPD).

    Returns:
        model (TOPRNet): trained neural network.
    """
    cfg = config.TrainingConfig()
    dataloader = get_dataloader()
    model = TOPRNet().cuda()
    loss_func = CustomLoss(model)
    optimizer = get_optimizer(model)
    lr_scheduler = None
    if cfg.use_scheduler:
        lr_scheduler = get_lr_scheduler(optimizer)
    summary(model)
    model.train()
    n_epoch = cfg.n_epoch + 1
    for epoch in tqdm(
        range(1, n_epoch),
        desc="Model training",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_func.forward(batch, mode)
            epoch_loss += loss.item()
            loss.backward()  # type: ignore[no-untyped-call]
            if cfg.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        epoch_loss = epoch_loss / len(dataloader)
        if epoch == 1 or epoch % cfg.report_interval == 0:
            print(f"\nEpoch {epoch}: loss = {epoch_loss:.12f} ")

    return model


def save_checkpoint(model: TOPRNet, mode: str) -> None:
    """Save checkpoint.

    Args:
        model (TOPRNet): trained neural network.
        mode (str): string to specfiy training mode (BPD or FPD).

    Returns:
       None.
    """
    path_cfg = config.PathConfig()
    model_dir = os.path.join(path_cfg.root_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    if mode == "bpd":
        model_file = os.path.join(model_dir, path_cfg.model_file + ".bpd.pth")
    else:
        model_file = os.path.join(model_dir, path_cfg.model_file + ".fpd.pth")
    torch.save(model.state_dict(), f=model_file)


def main() -> None:
    """Perform model training."""
    for cfg in (
        config.PathConfig(),
        config.PreProcessConfig(),
        config.FeatureConfig(),
        config.ModelConfig(),
        config.OptimizerConfig(),
        config.SchedulerConfig(),
        config.TrainingConfig(),
    ):
        print(cfg)

    model = training_loop("bpd")
    save_checkpoint(model, "bpd")

    model = training_loop("fpd")
    save_checkpoint(model, "fpd")


if __name__ == "__main__":
    main()
