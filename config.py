# -*- coding: utf-8 -*-
"""Config script of Two-stage Online/Offline Phase Reconstruction (TOPR).

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

from dataclasses import dataclass


@dataclass(frozen=True)
class PathConfig:
    """Dataclass for path config."""

    root_dir: str = "/work/tamamori/topr_lj/"
    data_dir: str = "data/"
    split_dir: str = "split/"
    feat_dir: str = "feat/"
    model_file: str = "model"
    demo_dir: str = "demo/"
    ltfat_dir: str = "/work/tamamori/ltfat-main"


@dataclass(frozen=True)
class PreProcessConfig:
    """Dataclass for preprocess."""

    n_dev: int = 300  # number of samples for development data
    n_eval: int = 300  # number of samples for evaluation data
    n_jobs: int = 6
    sec_per_split: float = 1.0  # seconds per split of a audio clip


@dataclass(frozen=True)
class FeatureConfig:
    """Dataclass for feature extraction."""

    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    window: str = "hann"
    n_fft: int = 1024


@dataclass(frozen=True)
class ModelConfig:
    """Dataclass for model definition."""

    n_lookahead: int = 0
    n_lookback: int = 3
    kernel_size: int = 5
    n_channels: int = 64


@dataclass(frozen=True)
class OptimizerConfig:
    """Dataclass for optimizer."""

    name: str = "RAdam"
    lr: float = 0.001
    weight_decay: float = 0.000001
    decoupled_weight_decay: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    """Dataclass for learning rate scheduler."""

    name: str = "CosineLRScheduler"
    warmup_t: int = 5
    t_initial: int = 15
    warmup_lr_init: float = 0.000001
    lr_min: float = 0.0002


@dataclass(frozen=True)
class TrainingConfig:
    """Dataclass for training."""

    n_epoch: int = 15
    n_batch: int = 32
    num_workers: int = 1
    report_interval: int = 1
    use_scheduler: bool = True
    use_grad_clip: bool = True
    grad_max_norm: float = 10.0


@dataclass(frozen=True)
class EvalConfig:
    """Dataclass for evaluation."""

    stoi_extended: bool = True  # True: extended STOI
    weighted_rpu: bool = False  # True: weighted RPU
    weight_power_rpu: float = 5.0  # power of weight
    weight_power: float = 2.0  # power of weight
    weight_gamma: float = 500.0  # scaler to weight
