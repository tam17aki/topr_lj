# -*- coding: utf-8 -*-
"""Proprocess script: resampling, trimming, and feature extraction.

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

import glob
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from hydra import compose, initialize
from progressbar import progressbar as prg
from pydub import AudioSegment
from scipy import signal


def make_filelist(cfg):
    """Make whole dataset into train, dev, and eval parts.

    Args:
        cfg (DictConfig): configuration in YAML format.
    """
    wav_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.data_dir, "orig")
    wav_list = glob.glob(wav_dir + "/*.wav")
    wav_list = random.sample(wav_list, len(wav_list))
    n_dev = cfg.preprocess.n_dev
    n_eval = cfg.preprocess.n_eval
    list_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir)
    os.makedirs(list_dir, exist_ok=True)

    for phase in ("train", "dev", "eval"):
        if phase == "train":
            file_list = wav_list[n_dev + n_eval :]
            file_name = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir, "train.list")
        elif phase == "dev":
            file_list = wav_list[:n_dev]
            file_name = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir, "dev.list")
        else:
            file_list = wav_list[n_dev : n_dev + n_eval]
            file_name = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir, "eval.list")

        with open(file_name, "w", encoding="utf-8") as file_handler:
            for wav_file in file_list:
                print(wav_file, file=file_handler)


def split_utterance(cfg):
    """Split utterances after resampling into segments.

    Args:
        cfg (DictConfig): configuration in YAML format.
    """
    out_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.data_dir, cfg.TOPR.split_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(
        os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir, "train.list"),
        "r",
        encoding="utf-8",
    ) as file_handler:
        wav_list = file_handler.read().splitlines()
    wav_list.sort()

    sec_per_split = cfg.preprocess.sec_per_split
    for wav_name in prg(
        wav_list, prefix="Split utterances: ", suffix=" ", redirect_stdout=False
    ):
        audio = AudioSegment.from_wav(wav_name)
        duration = math.floor(audio.duration_seconds)
        for i in range(0, int(duration // sec_per_split)):
            basename, ext = os.path.splitext(wav_name)
            split_fn = basename + "_" + str(i) + ext
            out_file = os.path.join(out_dir, os.path.basename(split_fn))
            split_audio = audio[i * 1000 : (i + sec_per_split) * 1000]
            # exclude samples less than 1.0 seconds
            if split_audio.duration_seconds > (sec_per_split - 0.01):
                split_audio.export(out_file, format="wav")


def _extract_feature(cfg, wav_file, feat_dir):
    """Perform feature extraction.

    Args:
        cfg (DictConfig): configuration in YAML format.
        wav_file (str): name of wav file.
        feat_dir (str): directory name for saving features.
    """
    audio, rate = sf.read(wav_file)
    if audio.dtype in [np.int16, np.int32]:
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float64)
    audio = audio.astype(np.float64)

    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
        hop=cfg.feature.hop_length,
        fs=rate,
        mfft=cfg.feature.n_fft,
    )
    stft_data = stfft.stft(audio)
    stft_data = stft_data.T  # transpose -> [n_frames, n_fft/2 +1]

    utt_id = os.path.splitext(os.path.basename(wav_file))[0]
    np.save(
        os.path.join(feat_dir, f"{utt_id}-feats_logmag.npy"),
        np.log(np.abs(stft_data)).astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        os.path.join(feat_dir, f"{utt_id}-feats_phase.npy"),
        np.angle(stft_data).astype(np.float32),
        allow_pickle=False,
    )


def extract_feature(cfg, phase):
    """Extract acoustic features.

    Args:
        cfg (DictConfig): configuration in YAML format.
        phase (str): handling dataset (training, development, or evaluation).
    """
    if phase == "train":
        wav_dir = os.path.join(
            cfg.TOPR.root_dir,
            cfg.TOPR.data_dir,
            cfg.TOPR.split_dir,
        )
        wav_list = glob.glob(wav_dir + "/*.wav")
        feat_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.feat_dir, "train")
    elif phase == "dev":
        with open(
            os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir, "dev.list"),
            "r",
            encoding="utf-8",
        ) as file_handler:
            wav_list = file_handler.read().splitlines()
        feat_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.feat_dir, "dev")
    else:
        with open(
            os.path.join(cfg.TOPR.root_dir, cfg.TOPR.list_dir, "eval.list"),
            "r",
            encoding="utf-8",
        ) as file_handler:
            wav_list = file_handler.read().splitlines()
        feat_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.feat_dir, "eval")

    wav_list.sort()
    os.makedirs(feat_dir, exist_ok=True)
    with ProcessPoolExecutor(cfg.preprocess.n_jobs) as executor:
        futures = [
            executor.submit(_extract_feature, cfg, wav_file, feat_dir)
            for wav_file in wav_list
        ]
        for future in prg(
            futures,
            prefix="Extract acoustic features: ",
            suffix=" ",
            redirect_stdout=False,
        ):
            future.result()  # return None


def main(cfg):
    """Perform preprocess."""
    make_filelist(cfg)
    split_utterance(cfg)
    extract_feature(cfg, "train")
    extract_feature(cfg, "dev")
    extract_feature(cfg, "eval")


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
