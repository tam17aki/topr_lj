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
from typing import TypedDict

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from pesq import pesq
from progressbar import progressbar as prg
from pystoi import stoi
from scipy import signal
from scipy.linalg import solve_banded
from scipy.sparse import csr_array
from torch.multiprocessing import set_start_method

import config
from model import TOPRNet


class WorkSpace(TypedDict):
    """Path dictionary for features."""

    ph_temp: npt.NDArray[np.float32]
    dwp: npt.NDArray[np.float32]
    fdd_coef: npt.NDArray[np.float32]
    coef: npt.NDArray[np.float32]
    rhs: npt.NDArray[np.float32]


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
    eval_cfg = config.EvalConfig()
    if model_cfg.n_lookahead == 0:
        if eval_cfg.weighted_rpu is True:
            wav_dir = os.path.join(
                path_cfg.root_dir, path_cfg.demo_dir, "online", "wRPU"
            )
        else:
            wav_dir = os.path.join(
                path_cfg.root_dir, path_cfg.demo_dir, "online", "RPU"
            )
    else:
        if eval_cfg.weighted_rpu is True:
            wav_dir = os.path.join(
                path_cfg.root_dir, path_cfg.demo_dir, "offline", "wRPU"
            )
        else:
            wav_dir = os.path.join(
                path_cfg.root_dir, path_cfg.demo_dir, "offline", "RPU"
            )
    return wav_dir


def get_wavname(basename: str) -> str:
    """Return filename of wavefile to be evaluated.

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        wav_file (str): filename of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_name = wav_name.split("_")[0][:-6]
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
    eval_wav = librosa.resample(
        eval_wav, orig_sr=rate, target_sr=16000, res_type="kaiser_best"
    )
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("_")[0][:-6]
    wav_dir = os.path.join(cfg.root_dir, cfg.data_dir, "orig")
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    reference = librosa.resample(
        reference, orig_sr=rate, target_sr=16000, res_type="kaiser_best"
    )
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return pesq(16000, reference, eval_wav)


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
    ref_wavname = ref_wavname.split("_")[0][:-6]
    wav_dir = os.path.join(path_cfg.root_dir, path_cfg.data_dir, "orig")
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return stoi(reference, eval_wav, rate, extended=eval_cfg.stoi_extended)


def compute_lsc(basename: str) -> np.float64:
    """Compute log-spectral convergence (LSC).

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        lsc (float): log-spectral convergence.
    """
    path_cfg = config.PathConfig()
    feat_cfg = config.FeatureConfig()
    eval_wav, _ = sf.read(get_wavname(basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("_")[0][:-6]
    wav_dir = os.path.join(path_cfg.root_dir, path_cfg.data_dir, "orig")
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
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


def bpd2tpd(bpd: npt.NDArray[np.float32], win_len: int, hop_len: int, n_frames: int):
    """Convert BPD to TPD.

    Args:
        bpd (ndarray): backward BPD. [K]
        win_len (int): length of analysis window.
        hop_len (int): length of window shift.
        n_frames (int): number of frames.

    Returns:
        tpd (ndarray): backward TPD. [K]
    """
    k = np.arange(0, win_len // 2 + 1)
    angle_freq = (2 * np.pi / win_len) * k * hop_len
    angle_freq = np.tile(np.expand_dims(angle_freq, 0), [n_frames, 1])
    tpd = bpd + angle_freq
    return tpd


@torch.no_grad()
def compute_1st_stage(
    model_tuple: tuple[TOPRNet, TOPRNet], logmag: torch.Tensor
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Estimate backward TPD and FPD from log-magnitude spectra.

    Args:
        model_tuple (Tuple): tuple of DNNs (nn.Module).
        logmag (Tensor): log-magnitude spectra. [T, L, K]

    Returns:
        tpd (ndarray): backward TPD. [T, K]
        fpd (ndarray): backward FPD. [T, K-1]
    """
    model_cfg = config.ModelConfig()
    feat_cfg = config.FeatureConfig()
    logmag = logmag.unfold(
        1, model_cfg.n_lookback + model_cfg.n_lookahead + 1, 1
    )  # [1, T, K, L]
    _, n_frame, n_fbin, width = logmag.shape  # [1, T, K, L]
    logmag = logmag.reshape(-1, n_fbin, width)  # [T, K, L]
    logmag = logmag.transpose(2, 1)  # [T, L, K]
    model_bpd, model_fpd = model_tuple  # DNNs
    bpd = model_bpd(logmag)  # [T, 1, K]
    fpd = model_fpd(logmag)  # [T, 1, K]
    bpd = bpd.cpu().detach().numpy().copy().squeeze()  # [T, K]
    fpd = fpd.cpu().detach().numpy().copy().squeeze()  # [T, K]
    tpd = bpd2tpd(bpd, feat_cfg.win_length, feat_cfg.hop_length, n_frame)
    return tpd, fpd[:, :-1]  # [T, K], [T, K-1]


def get_band_coef(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return band tridiagonal elements of coef matrix.

    Args:
        matrix (ndarray): band tridiagonal matrix.

    Returns:
        band_elem (ndarray): band tridiagonal elements (upper, diag, and lower).
    """
    upper = np.diag(matrix, 1)
    upper = np.concatenate((np.array([0]), upper))
    lower = np.diag(matrix, -1)
    lower = np.concatenate((lower, np.array([0])))
    band_elem = np.concatenate(
        (upper.reshape(1, -1), np.diag(matrix).reshape(1, -1), lower.reshape(1, -1))
    )
    return band_elem


def wrap_phase(phase):
    """Compute wrapped phase.

    Args:
        phase (ndarray): phase spectrum.

    Returns:
        wrapped phase (ndarray).
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def compute_rpu(
    ifreq: npt.NDArray[np.float32],
    grd: npt.NDArray[np.float32],
    magnitude: npt.NDArray[np.float32],
    weighted_rpu: bool,
    weight_power: float,
) -> npt.NDArray[np.float32]:
    """Reconstruct phase by Recurrent Phase Unwrapping (RPU).

    This function performs phase reconstruction via RPU.

    Y. Masuyama, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada,
    Phase reconstruction based on recurrent phase unwrapping with deep neural
    networks, IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2020.

    For weighted RPU, see:

    N. B. Thien, Y. Wakabayashi, K. Iwai and T. Nishiura,
    Inter-Frequency Phase Difference for Phase Reconstruction Using Deep Neural
    Networks and Maximum Likelihood, in IEEE/ACM Transactions on Audio,
    Speech, and Language Processing, vol. 31, pp. 1667-1680, 2023.

    Args:
        ifreq (ndarray): instantaneous frequency. [T-1, K]
        grd   (ndarray): group delay. [T, K-1]
        magnitude (ndarray): magnitude spectrum. [T, K]
        weighted_rpu (bool): flag to apply weighted RPU.
        weight_power (float): power to weight.

    Returns:
        phase (ndarray): reconstructed phase. [T, K]
    """
    n_frame, n_feats = magnitude.shape
    phase = np.zeros_like(magnitude)
    fd_mat = (  # frequency-directional differential operator (matrix)
        -np.triu(np.ones((n_feats - 1, n_feats)), 1)
        + np.triu(np.ones((n_feats - 1, n_feats)), 2)
        + np.eye(n_feats - 1, n_feats)
    ).astype(np.float32)
    fd_mat = csr_array(fd_mat)
    var = WorkSpace(
        {
            "ph_temp": np.empty(1).astype(np.float32),
            "dwp": np.empty(1).astype(np.float32),
            "fdd_coef": np.empty(1).astype(np.float32),
            "coef": np.empty(1).astype(np.float32),
            "rhs": np.empty(1).astype(np.float32),
        }
    )

    for k in range(1, n_feats):
        phase[0, k] = phase[0, k - 1] - grd[0, k - 1]
    if weighted_rpu is False:
        var["coef"] = fd_mat.T @ fd_mat + np.eye(n_feats).astype(np.float32)
        var["coef"] = get_band_coef(var["coef"])
        for i in range(1, n_frame):
            var["ph_temp"] = wrap_phase(phase[i - 1, :]) + ifreq[i - 1, :]
            var["dwp"] = fd_mat @ var["ph_temp"]
            grd_new = var["dwp"] + wrap_phase(grd[i, :] - var["dwp"])
            var["rhs"] = var["ph_temp"] + fd_mat.T @ grd_new
            phase[i, :] = solve_banded((1, 1), var["coef"], var["rhs"])
    else:
        for i in range(1, n_frame):
            w_ifreq = magnitude[i - 1, :] ** weight_power
            w_grd = magnitude[i, :-1] ** weight_power
            var["fdd_coef"] = fd_mat.T * w_grd
            var["coef"] = np.diag(w_ifreq).astype(np.float32) + var["fdd_coef"] @ fd_mat
            var["coef"] = get_band_coef(var["coef"])
            var["ph_temp"] = wrap_phase(phase[i - 1, :]) + ifreq[i - 1, :]
            var["dwp"] = fd_mat @ var["ph_temp"]
            grd_new = var["dwp"] + wrap_phase(grd[i, :] - var["dwp"])
            var["rhs"] = w_ifreq * var["ph_temp"] + var["fdd_coef"] @ grd_new
            phase[i, :] = solve_banded((1, 1), var["coef"], var["rhs"])
    return phase


def _reconst_waveform(model_tuple: tuple[TOPRNet, TOPRNet], logmag_path: str) -> None:
    """Reconstruct audio waveform only from the magnitude spectra.

    Notice that the instantaneous frequency and group delay are estimated
    by the 1st stage of TOPR, respectively.

    The phase spectrum is reconstruced via RPU.

    Args:
        model_tuple (Tuple): tuple of DNN params (nn.Module).
        logmag_path (str): path to the log-magnitude spectrum.

    Returns:
        None.
    """
    model_cfg = config.ModelConfig()
    feat_cfg = config.FeatureConfig()
    eval_cfg = config.EvalConfig()
    logmag = np.load(logmag_path)  # [T, K]
    magnitude = np.exp(logmag)  # [T, K]

    # estimate TPD and FPD from log-magnitude spectra.
    logmag = np.pad(
        logmag, ((model_cfg.n_lookback, model_cfg.n_lookahead), (0, 0)), "constant"
    )
    logmag = torch.from_numpy(logmag).float().unsqueeze(0).cuda()  # [1, T+L+1, K]
    tpd, fpd = compute_1st_stage(model_tuple, logmag)

    # reconstruct phase spectra by using instantaneous frequency (= TPD) and
    # group delay (= negative FPD).
    phase = compute_rpu(
        tpd, -fpd, magnitude, eval_cfg.weighted_rpu, eval_cfg.weight_power_rpu
    )

    # reconstruct audio waveform
    reconst_spec = magnitude * np.exp(1j * phase)  # [T, K]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(feat_cfg.window, feat_cfg.win_length),
        hop=feat_cfg.hop_length,
        fs=feat_cfg.sample_rate,
        mfft=feat_cfg.n_fft,
    )
    audio = stfft.istft(reconst_spec.T)
    wav_file = get_wavname(os.path.basename(logmag_path))
    sf.write(wav_file, audio, feat_cfg.sample_rate)


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
        for future in prg(
            futures, prefix="Reconstruct waveform: ", suffix=" ", redirect_stdout=False
        ):
            future.result()  # return None


def compute_obj_scores(logmag_list: list[str]) -> dict[str, list[float]]:
    """Compute objective scores; PESQ, STOI and LSC.

    Args:
        logmag_list (list): list of path to the log-magnitude spectrum.

    Returns:
        score_dict (dict): dictionary of objective score lists.
    """
    score_dict = {"pesq": [], "stoi": [], "lsc": []}
    for logmag_path in prg(
        logmag_list,
        prefix="Compute objective scores: ",
        suffix=" ",
        redirect_stdout=False,
    ):
        score_dict["pesq"].append(compute_pesq(os.path.basename(logmag_path)))
        score_dict["stoi"].append(compute_stoi(os.path.basename(logmag_path)))
        score_dict["lsc"].append(compute_lsc(os.path.basename(logmag_path)))
    return score_dict


def aggregate_scores(score_dict: dict[str, list[float]], score_dir: str) -> None:
    """Aggregate objective evaluation scores.

    Args:
        score_dict (dict): dictionary of objective score lists.
        score_dir (str): dictionary name of objective score files.

    Returns:
        None.
    """
    model_cfg = config.ModelConfig()
    eval_cfg = config.EvalConfig()
    for score_type, score_list in score_dict.items():
        if model_cfg.n_lookahead == 0:
            if eval_cfg.weighted_rpu is True:
                out_filename = f"{score_type}_score_wRPU_online.txt"
            else:
                out_filename = f"{score_type}_score_RPU_online.txt"
        else:
            if eval_cfg.weighted_rpu is True:
                out_filename = f"{score_type}_score_wRPU_offline.txt"
            else:
                out_filename = f"{score_type}_score_RPU_offline.txt"
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
    logmag_list = []
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
