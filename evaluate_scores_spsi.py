# -*- coding: utf-8 -*-
"""Evaluation script for sound quality of SPSI.

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

import librosa
import numpy as np
import soundfile as sf
from hydra import compose, initialize
from oct2py import octave
from omegaconf import DictConfig
from pesq import pesq
from progressbar import progressbar as prg
from pystoi import stoi
from scipy import signal


def get_wavdir(cfg):
    """Return dirname of wavefile to be evaluated.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        wav_dir (str): dirname of wavefile.
    """
    wav_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.demo_dir, "online", "SPSI")
    return wav_dir


def get_wavname(cfg, basename):
    """Return filename of wavefile to be evaluated.

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        wav_file (str): filename of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_dir = get_wavdir(cfg)
    wav_file = os.path.join(wav_dir, wav_name + ".wav")
    return wav_file


def compute_pesq(cfg, wav_path):
    """Compute PESQ and wideband PESQ.

    Args:
        cfg (DictConfig): configuration.
        wav_path (str): pathname of wavefile for evaluation.

    Returns:
        float: PESQ (or wideband PESQ).
    """
    eval_wav, rate = sf.read(get_wavname(cfg, os.path.basename(wav_path)))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    reference, rate = sf.read(wav_path)
    reference = librosa.resample(reference, orig_sr=rate, target_sr=16000)
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return pesq(16000, reference, eval_wav)


def compute_stoi(cfg, wav_path):
    """Compute STOI or extended STOI (ESTOI).

    Args:
        cfg (DictConfig): configuration.
        wav_path (str): pathname of wavefile for evaluation.

    Returns:
        float: STOI (or ESTOI).
    """
    eval_wav, rate = sf.read(get_wavname(cfg, os.path.basename(wav_path)))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    reference, rate = sf.read(wav_path)
    reference = librosa.resample(reference, orig_sr=rate, target_sr=16000)
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return stoi(reference, eval_wav, rate, extended=cfg.demo.stoi_extended)


def compute_lsc(cfg, wav_path):
    """Compute log-spectral convergence (LSC).

    Args:
        cfg (DictConfig): configuration.
        wav_path (str): pathname of wavefile for evaluation.

    Returns:
        float: log-spectral convergence.
    """
    eval_wav, rate = sf.read(get_wavname(cfg, os.path.basename(wav_path)))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    reference, rate = sf.read(wav_path)
    reference = librosa.resample(reference, orig_sr=rate, target_sr=16000)
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
        hop=cfg.feature.hop_length,
        fs=rate,
        mfft=cfg.feature.n_fft,
    )
    ref_abs = np.abs(stfft.stft(reference))
    eval_abs = np.abs(stfft.stft(eval_wav))
    lsc = np.linalg.norm(ref_abs - eval_abs)
    lsc = lsc / np.linalg.norm(ref_abs)
    lsc = 20 * np.log10(lsc)
    return lsc


def reconst_waveform(cfg, wav_list):
    """Reconstruct audio waveform only from the magnitude spectrum.

    Args:
        cfg (DictConfig): configuration.
        wav_list (list): list of path to wav file.

    Returns:
        None.
    """
    for wav_path in prg(
        wav_list, prefix="Reconstruct waveform: ", suffix=" ", redirect_stdout=False
    ):
        audio, _ = sf.read(wav_path)
        stfft = signal.ShortTimeFFT(
            win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
            hop=cfg.feature.hop_length,
            fs=cfg.feature.sample_rate,
            mfft=cfg.feature.n_fft,
        )
        magnitude = np.abs(stfft.stft(audio))
        reconst_spec = octave.spsi(
            magnitude, cfg.feature.hop_length, cfg.feature.win_length
        )
        audio = stfft.istft(reconst_spec)
        wav_file = get_wavname(cfg, os.path.basename(wav_path))
        sf.write(wav_file, audio, cfg.feature.sample_rate)


def compute_obj_scores(cfg, wav_list):
    """Compute objective evaluation scores; PESQ, STOI and LSC.

    Args:
        cfg (DictConfig): configuration.
        wav_list (list): list of path to wav file.

    Returns:
        score_dict (dict): dictionary of objective score lists.
    """
    score_dict = {"pesq": [], "stoi": [], "lsc": []}
    for wav_path in prg(
        wav_list, prefix="Compute objective scores: ", suffix=" ", redirect_stdout=False
    ):
        score_dict["pesq"].append(compute_pesq(cfg, wav_path))
        score_dict["stoi"].append(compute_stoi(cfg, wav_path))
        score_dict["lsc"].append(compute_lsc(cfg, wav_path))
    return score_dict


def aggregate_scores(score_dict, score_dir):
    """Aggregate objective evaluation scores.

    Args:
        score_dict (dict): dictionary of objective score lists.
        score_dir (str): dictionary name of objective score files.

    Returns:
        None.
    """
    for score_type, score_list in score_dict.items():
        out_filename = f"{score_type}_score_SPSI.txt"
        out_filename = os.path.join(score_dir, out_filename)
        with open(out_filename, mode="w", encoding="utf-8") as file_handler:
            for score in score_list:
                file_handler.write(f"{score}\n")
        score_array = np.array(score_list)
        print(
            f"{score_type}: "
            f"mean={np.mean(score_array):.6f}, "
            f"median={np.median(score_array):.6f}, "
            f"std={np.std(score_array):.6f}, "
            f"max={np.max(score_array):.6f}, "
            f"min={np.min(score_array):.6f}"
        )


def main(cfg: DictConfig):
    """Perform evaluation."""
    # initialization for octave
    octave.addpath(octave.genpath(config.TOPR.ltfat_dir))
    octave.ltfatstart(0)
    octave.phaseretstart(0)

    # setup directory
    wav_dir = get_wavdir(cfg)  # dirname for reconstructed wav files
    os.makedirs(wav_dir, exist_ok=True)
    score_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.score_dir)
    os.makedirs(score_dir, exist_ok=True)

    # reconstruct phase and waveform
    with open(
        os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir, "eval.list"),
        "r",
        encoding="utf-8",
    ) as file_handler:
        wav_list = file_handler.read().splitlines()
    reconst_waveform(cfg, wav_list)

    # compute objective scores
    score_dict = compute_obj_scores(cfg, wav_list)

    # aggregate objective scores
    aggregate_scores(score_dict, score_dir)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
