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
from oct2py import octave
from pesq import pesq
from pystoi import stoi
from scipy import signal
from tqdm import tqdm

import config


def get_wavdir() -> str:
    """Return dirname of wavefile to be evaluated.

    Args:
        None.

    Returns:
        wav_dir (str): dirname of wavefile.
    """
    cfg = config.PathConfig()
    wav_dir = os.path.join(cfg.root_dir, cfg.demo_dir, "online", "SPSI")
    return wav_dir


def get_wavname(basename: str) -> str:
    """Return filename of wavefile to be evaluated.

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        wav_file (str): filename of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_dir = get_wavdir()
    wav_file = os.path.join(wav_dir, wav_name + ".wav")
    return wav_file


def compute_pesq(wav_path: str) -> float:
    """Compute PESQ and wideband PESQ.

    Args:
        wav_path (str): pathname of wavefile for evaluation.

    Returns:
        float: PESQ (or wideband PESQ).
    """
    eval_wav, rate = sf.read(get_wavname(os.path.basename(wav_path)))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    reference, rate = sf.read(wav_path)
    reference = librosa.resample(reference, orig_sr=rate, target_sr=16000)
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    return float(pesq(16000, reference, eval_wav))


def compute_stoi(wav_path: str) -> float:
    """Compute STOI or extended STOI (ESTOI).

    Args:
        wav_path (str): pathname of wavefile for evaluation.

    Returns:
        float: STOI (or ESTOI).
    """
    cfg = config.EvalConfig()
    eval_wav, rate = sf.read(get_wavname(os.path.basename(wav_path)))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    reference, rate = sf.read(wav_path)
    reference = librosa.resample(reference, orig_sr=rate, target_sr=16000)
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    return float(stoi(reference, eval_wav, rate, extended=cfg.stoi_extended))


def compute_lsc(wav_path: str) -> np.float64:
    """Compute log-spectral convergence (LSC).

    Args:
        wav_path (str): pathname of wavefile for evaluation.

    Returns:
        float: log-spectral convergence.
    """
    cfg = config.FeatureConfig()
    eval_wav, rate = sf.read(get_wavname(os.path.basename(wav_path)))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    reference, rate = sf.read(wav_path)
    reference = librosa.resample(reference, orig_sr=rate, target_sr=16000)
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.window, cfg.win_length),
        hop=cfg.hop_length,
        fs=rate,
        mfft=cfg.n_fft,
    )
    ref_abs = np.abs(stfft.stft(reference))
    eval_abs = np.abs(stfft.stft(eval_wav))
    lsc = np.linalg.norm(ref_abs - eval_abs)
    lsc = lsc / np.linalg.norm(ref_abs)
    lsc = 20 * np.log10(lsc)
    return lsc


def reconst_waveform(wav_list: list[str]) -> None:
    """Reconstruct audio waveform only from the magnitude spectrum.

    Args:
        wav_list (list): list of path to wav file.

    Returns:
        None.
    """
    cfg = config.FeatureConfig()
    for wav_path in tqdm(
        wav_list,
        desc="Reconstruct waveform",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        audio, _ = sf.read(wav_path)
        stfft = signal.ShortTimeFFT(
            win=signal.get_window(cfg.window, cfg.win_length),
            hop=cfg.hop_length,
            fs=cfg.sample_rate,
            mfft=cfg.n_fft,
        )
        magnitude = np.abs(stfft.stft(audio))
        reconst_spec = octave.spsi(magnitude, cfg.hop_length, cfg.win_length)
        audio = stfft.istft(reconst_spec)
        wav_file = get_wavname(os.path.basename(wav_path))
        sf.write(wav_file, audio, cfg.sample_rate)


def compute_obj_scores(wav_list: list[str]) -> dict[str, list[np.float64 | float]]:
    """Compute objective evaluation scores; PESQ, STOI and LSC.

    Args:
        wav_list (list): list of path to wav file.

    Returns:
        score_dict (dict): dictionary of objective score lists.
    """
    score_dict: dict[str, list[np.float64 | float]] = {
        "pesq": [],
        "stoi": [],
        "lsc": [],
    }
    for wav_path in tqdm(
        wav_list,
        desc="Compute objective scores: ",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        score_dict["pesq"].append(compute_pesq(wav_path))
        score_dict["stoi"].append(compute_stoi(wav_path))
        score_dict["lsc"].append(compute_lsc(wav_path))
    return score_dict


def aggregate_scores(
    score_dict: dict[str, list[np.float64 | float]], score_dir: str
) -> None:
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


def main() -> None:
    """Perform evaluation."""
    # initialization for octave
    cfg = config.PathConfig()
    octave.addpath(octave.genpath(cfg.ltfat_dir))
    octave.ltfatstart(0)
    octave.phaseretstart(0)

    # setup directory
    wav_dir = get_wavdir()  # dirname for reconstructed wav files
    os.makedirs(wav_dir, exist_ok=True)
    score_dir = os.path.join(cfg.root_dir, "score")
    os.makedirs(score_dir, exist_ok=True)

    # reconstruct phase and waveform
    with open(
        os.path.join(cfg.root_dir, "list", "eval.list"), "r", encoding="utf-8"
    ) as file_handler:
        wav_list = file_handler.read().splitlines()
    reconst_waveform(wav_list)

    # compute objective scores
    score_dict = compute_obj_scores(wav_list)

    # aggregate objective scores
    aggregate_scores(score_dict, score_dir)


if __name__ == "__main__":
    main()
