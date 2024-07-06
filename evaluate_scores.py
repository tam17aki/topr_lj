# -*- coding: utf-8 -*-
"""Evaluation script for sound quality based on PESQ, STOI and LSC.

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
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from scipy import signal
from scipy.sparse import csr_array, diags_array
from scipy.sparse.linalg import spsolve
from torch.multiprocessing import set_start_method
from tqdm import tqdm

import config
from model import TOPRNet


def load_checkpoint() -> tuple[TOPRNet, TOPRNet]:
    """Load checkpoint.

    Args:
        None.

    Returns:
        model_bpd (nn.Module): DNNs to estimate BPD.
        model_fpd (nn.Module): DNNs to estimate FPD.
    """
    cfg = config.PathConfig()
    model_dir = os.path.join(cfg.root_dir, "model")
    model_bpd = TOPRNet().cuda()
    model_file = os.path.join(model_dir, cfg.model_file + ".bpd.pth")
    checkpoint = torch.load(model_file)
    model_bpd.load_state_dict(checkpoint)

    model_fpd = TOPRNet().cuda()
    model_file = os.path.join(model_dir, cfg.model_file + ".fpd.pth")
    checkpoint = torch.load(model_file)
    model_fpd.load_state_dict(checkpoint)
    return model_bpd, model_fpd


def get_wavdir() -> str:
    """Return dirname of wavefile to be evaluated.

    Args:
        None.

    Returns:
        wav_dir (str): dirname of wavefile.
    """
    path_cfg = config.PathConfig()
    model_cfg = config.ModelConfig()
    if model_cfg.n_lookahead == 0:
        wav_dir = os.path.join(path_cfg.root_dir, path_cfg.demo_dir, "online", "TOPR")
    else:
        wav_dir = os.path.join(path_cfg.root_dir, path_cfg.demo_dir, "offline", "TOPR")
    return wav_dir


def get_wavname(basename: str) -> str:
    """Return filename of wavefile to be evaluated.

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        wav_file (str): filename of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_name = wav_name.split("_")[0][:-6]  # remove '_logmag'
    wav_dir = get_wavdir()
    wav_file = os.path.join(wav_dir, wav_name + ".wav")
    return wav_file


def compute_pesq(basename: str) -> float:
    """Compute PESQ and wideband PESQ.

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: PESQ (or wideband PESQ).
    """
    cfg = config.PathConfig()
    eval_wav, rate = sf.read(get_wavname(basename))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("_")[0][:-6]  # remove '_logmag'
    wav_dir = os.path.join(cfg.root_dir, cfg.data_dir, "orig")
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    reference = librosa.resample(y=reference, orig_sr=rate, target_sr=16000)
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    return float(pesq(16000, reference, eval_wav))


def compute_stoi(basename: str) -> float:
    """Compute STOI or extended STOI (ESTOI).

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: STOI (or ESTOI).
    """
    path_cfg = config.PathConfig()
    eval_cfg = config.EvalConfig()
    eval_wav, _ = sf.read(get_wavname(basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("_")[0][:-6]  # remove '_logmag'
    wav_dir = os.path.join(path_cfg.root_dir, path_cfg.data_dir, "orig")
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    return float(stoi(reference, eval_wav, rate, extended=eval_cfg.stoi_extended))


def compute_lsc(basename: str) -> np.float64:
    """Compute log-spectral convergence (LSC).

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        lsc (float64): log-spectral convergence.
    """
    path_cfg = config.PathConfig()
    feat_cfg = config.FeatureConfig()
    eval_wav, _ = sf.read(get_wavname(basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("_")[0][:-6]  # remove '_logmag'
    wav_dir = os.path.join(path_cfg.root_dir, path_cfg.data_dir, "orig")
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(feat_cfg.window, feat_cfg.win_length),
        hop=feat_cfg.hop_length,
        fs=rate,
        mfft=feat_cfg.n_fft,
    )
    ref_abs = np.abs(stfft.stft(reference))
    eval_abs = np.abs(stfft.stft(eval_wav))
    lsc = np.linalg.norm(ref_abs - eval_abs)
    lsc = lsc / np.linalg.norm(ref_abs)
    lsc = 20 * np.log10(lsc)
    return lsc


def bpd2tpd(
    bpd: npt.NDArray[np.float32], win_len: int, hop_len: int
) -> npt.NDArray[np.float32]:
    """Convert BPD to TPD.

    Args:
        bpd (ndarray): BPD.
        win_len (int): length of analysis window.
        hop_len (int): length of window shift.

    Returns:
        tpd (ndarray): TPD.
    """
    k = np.arange(0, win_len // 2 + 1)
    angle_freq = (2 * np.pi / win_len) * k * hop_len
    tpd = bpd + angle_freq
    return tpd


@torch.no_grad()
def compute_1st_stage(
    model_tuple: tuple[TOPRNet, TOPRNet], logmag: torch.Tensor
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Estimate TPD and FPD from log-magnitude spectra.

    Args:
        model_tuple (tuple): tuple of DNN params (nn.Module).
        logmag (ndarray): log magnitude spectrum. [1, L, K]

    Returns:
        tpd (ndarray): TPD. [K]
        fpd (ndarray): FPD. [K-1]
    """
    cfg = config.FeatureConfig()
    model_bpd, model_fpd = model_tuple  # DNNs
    bpd = model_bpd(logmag)  # [1, 1, K]
    fpd = model_fpd(logmag)  # [1, 1, K]
    bpd = bpd.cpu().detach().numpy().copy().squeeze()
    fpd = fpd.cpu().detach().numpy().copy().squeeze()
    fpd = fpd[:-1]
    tpd = bpd2tpd(bpd, cfg.win_length, cfg.hop_length)
    return tpd, fpd


def compute_2nd_stage(
    phase_prev: npt.NDArray[np.float32],
    pd_tuple: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
    mag_cur: npt.NDArray[np.float32],
    mag_prev: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Reconstruct phase spectrum.

    Args:
        phase_prev (ndarray): phase spectrum at the previous frame. [K]
        pd_tuple (Tuple): tuple of TPD and FPD (ndarray).
        mag_cur (ndarray): magnitude spectrum at the current frame. [K]
        mag_prev (ndarray): magnitude spectrum at the previous frame. [K]

    Returns:
        phase (ndarray): reconstructed phase spectrum at the current frame. [K]
    """
    cfg = config.EvalConfig()
    tpd, fpd = pd_tuple
    n_fbin = mag_cur.shape[0]

    # complex ratios
    ratio_u = mag_cur[1:] / mag_cur[:-1]  # [K-1]
    ratio_u = ratio_u * np.exp(1j * fpd)  # [K-1]  Eqs. (37) and (41)

    # weight matrix (diagonal)
    lambda_vec = (mag_cur * mag_prev) ** cfg.weight_power
    gamma_vec = cfg.weight_gamma * ((mag_cur[1:] * mag_cur[:-1]) ** cfg.weight_power)

    d_mat = csr_array(
        (
            np.append(-1.0 * ratio_u, np.ones(n_fbin - 1)),  # data
            (
                list(range(n_fbin - 1)) + list(range(n_fbin - 1)),
                list(range(n_fbin - 1)) + list(range(1, n_fbin)),
            ),  # (row_ind, col_ind)
        ),
        shape=(n_fbin - 1, n_fbin),
        dtype=np.complex64,
    )  # Eqs. (44) and (45)
    coef = (
        diags_array(lambda_vec, format="csr")
        + d_mat.T.tocsr() @ diags_array(gamma_vec, format="csr") @ d_mat
    )
    rhs = lambda_vec * mag_cur * np.exp(1j * (phase_prev + tpd))
    phase = np.angle(spsolve(coef, rhs))
    return phase


def reconst_phase(
    model_tuple: tuple[TOPRNet, TOPRNet],
    logmag: npt.NDArray[np.float32],
    magnitude: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Reconstruct phase spectrum by TOPR algorithm.

    Y. Masuyama, K. Yatabe, K. Nagatomo and Y. Oikawa,
    "Online Phase Reconstruction via DNN-Based Phase Differences Estimation,"
    in IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    vol. 31, pp. 163-176, 2023, doi: 10.1109/TASLP.2022.3221041.

    Args:
        model_tuple (Tuple): tuple of DNNs (nn.Module).
        logmag (ndarray): log-magnitude spectrum (zero padded). [T, K]
        magnitude (ndarray): magnitude spectrum. [T, K]

    Returns:
        phase (ndarray): reconstruced phase. [T, K]
    """
    cfg = config.ModelConfig()
    logmag = np.pad(logmag, ((cfg.n_lookback, cfg.n_lookahead), (0, 0)), "constant")
    logmag_tensor = torch.tensor(logmag).float().unsqueeze(0).cuda()  # [1, T+L+1, K]
    n_frame, n_fbin = magnitude.shape
    n_lookback = cfg.n_lookback
    n_lookahead = cfg.n_lookahead

    phase = np.zeros((n_frame, n_fbin)).astype(np.float32)  # [T, K]
    _, fpd = compute_1st_stage(
        model_tuple, logmag_tensor[:, : n_lookback + n_lookahead + 1, :]
    )
    for k in range(1, n_fbin):
        phase[0, k] = phase[0, k - 1] + fpd[k - 1]
    for i in range(1, n_frame):
        tpd, fpd = compute_1st_stage(
            model_tuple, logmag_tensor[:, i : i + n_lookback + n_lookahead + 1, :]
        )  # Eqs. (29), (30)
        phase[i, :] = compute_2nd_stage(
            phase[i - 1, :], (tpd, fpd), magnitude[i, :], magnitude[i - 1, :]
        )  # Eq. (31)
    return phase


def _reconst_waveform(model_tuple: tuple[TOPRNet, TOPRNet], logmag_path: str) -> None:
    """Reconstruct audio waveform only from the magnitude spectrum.

    Args:
        model_tuple (Tuple): tuple of DNN params (nn.Module).
        logmag_path (str): path to the log-magnitude spectrum.

    Returns:
        None.
    """
    cfg = config.FeatureConfig()
    logmag = np.load(logmag_path)  # [T, K]
    magnitude = np.exp(logmag)  # [T, K]
    phase = reconst_phase(model_tuple, logmag, magnitude)  # [T, K]
    reconst_spec = magnitude * np.exp(1j * phase)  # [T, K]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.window, cfg.win_length),
        hop=cfg.hop_length,
        fs=cfg.sample_rate,
        mfft=cfg.n_fft,
    )
    audio = stfft.istft(reconst_spec.T)
    wav_file = get_wavname(os.path.basename(logmag_path))
    sf.write(wav_file, audio, cfg.sample_rate)


def reconst_waveform(
    model_tuple: tuple[TOPRNet, TOPRNet], logmag_list: list[str]
) -> None:
    """Reconstruct audio waveforms in parallel.

    Args:
        model_tuple (Tuple): tuple of DNN params (nn.Module).
        logmag_list (list): list of path to the log-magnitude spectrum.

    Returns:
        None.
    """
    cfg = config.PreProcessConfig()
    set_start_method("spawn")
    with ProcessPoolExecutor(cfg.n_jobs) as executor:
        futures = [
            executor.submit(_reconst_waveform, model_tuple, logmag_path)
            for logmag_path in logmag_list
        ]
        for future in tqdm(
            futures,
            desc="Reconstruct waveform",
            bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
            " Elapsed Time: {elapsed} ETA: {remaining} ",
            ascii=" #",
        ):
            future.result()  # return None


def compute_obj_scores(logmag_list: list[str]) -> dict[str, list[np.float64 | float]]:
    """Compute objective scores; PESQ, STOI and LSC.

    Args:
        logmag_list (list): list of path to the log-magnitude spectrum.

    Returns:
        score_dict (dict): dictionary of objective score lists.
    """
    score_dict: dict[str, list[np.float64 | float]] = {
        "pesq": [],
        "stoi": [],
        "lsc": [],
    }
    for logmag_path in tqdm(
        logmag_list,
        desc="Compute objective scores",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        score_dict["pesq"].append(compute_pesq(os.path.basename(logmag_path)))
        score_dict["stoi"].append(compute_stoi(os.path.basename(logmag_path)))
        score_dict["lsc"].append(compute_lsc(os.path.basename(logmag_path)))
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
    cfg = config.ModelConfig()
    for score_type, score_list in score_dict.items():
        if cfg.n_lookahead == 0:
            out_filename = f"{score_type}_score_TOPR_online.txt"
        else:
            out_filename = f"{score_type}_score_TOPR_offline.txt"
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


def load_logmag() -> list[str]:
    """Load file paths for log-magnitude spectrogram.

    Args:
        None.

    Returns:
        logmag_list (list): list of file path for log-magnitude spectrogram.
    """
    cfg = config.PathConfig()
    feat_dir = os.path.join(cfg.root_dir, cfg.feat_dir, "eval")
    logmag_list: list[str] = []
    with open(
        os.path.join(cfg.root_dir, "list", "eval.list"), "r", encoding="utf-8"
    ) as file_handler:
        wav_list = file_handler.read().splitlines()
    for wav_name in wav_list:
        basename, _ = os.path.splitext(os.path.basename(wav_name))
        logmag_list.append(os.path.join(feat_dir, basename + "-feats_logmag.npy"))
    logmag_list.sort()
    return logmag_list


def main() -> None:
    """Perform evaluation."""
    # setup directory
    cfg = config.PathConfig()
    wav_dir = get_wavdir()
    os.makedirs(wav_dir, exist_ok=True)
    score_dir = os.path.join(cfg.root_dir, "score")
    os.makedirs(score_dir, exist_ok=True)

    # load DNN parameters
    model_bpd, model_fpd = load_checkpoint()
    model_bpd.cuda()
    model_fpd.cuda()
    model_bpd.eval()
    model_fpd.eval()

    # load list of file paths for log-magnitude spectrogram.
    logmag_list = load_logmag()

    # reconstruct phase and waveform
    reconst_waveform((model_bpd, model_fpd), logmag_list)

    # compute objective scores
    score_dict = compute_obj_scores(logmag_list)

    # aggregate objective scores
    aggregate_scores(score_dict, score_dir)


if __name__ == "__main__":
    main()
