"""The vocoder's log-mel spectrogram, and A-weighted loudness on the same frames.

LogMel reproduces PitchAdjustableMelSpectrogram (key_shift = 0) and the log
compression from openvpi/SingingVocoders (utils/wav2mel.py,
training/nsf_HiFigan_task.py), which PC-NSF-HiFiGAN was trained on:

    MIT License, Copyright (c) 2023 Team OpenVPI

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions: The above copyright notice and this permission
    notice shall be included in all copies or substantial portions of the
    Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

Framing: reflect padding of (n_fft - hop) / 2 on each side, no centring, so
N samples give N // hop frames and frame j spans samples
[j * hop - 768, j * hop + 1280). The vocoder pairs frame j with the F0 at
time j * hop / sr (see frontend.f0.mel_frame_times).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

from config import FrontendConfig


class LogMel(nn.Module):
    mel_basis: torch.Tensor
    window: torch.Tensor

    def __init__(self, cfg: FrontendConfig = FrontendConfig()):
        super().__init__()
        self.cfg = cfg
        basis = librosa_mel(
            sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
        )  # slaney scale and norm, as in the vocoder
        self.register_buffer(
            "mel_basis", torch.from_numpy(basis).float(), persistent=False
        )
        self.register_buffer(
            "window", torch.hann_window(cfg.win_length), persistent=False
        )
        self.pad = (
            (cfg.n_fft - cfg.hop_length) // 2,
            (cfg.n_fft - cfg.hop_length + 1) // 2,
        )

    def magnitude(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, N] -> |STFT| [B, n_fft // 2 + 1, N // hop]
        x = F.pad(audio.unsqueeze(1), self.pad, mode="reflect").squeeze(1)
        return torch.stft(
            x,
            self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, N] at cfg.sr -> natural-log mel [B, n_mels, N // hop]
        mel = self.mel_basis @ self.magnitude(audio)
        return torch.log(mel.clamp(min=self.cfg.log_mel_floor))


def a_weighting_db(freqs: np.ndarray) -> np.ndarray:
    # IEC 61672-1 A-weighting, 0 dB at 1 kHz.
    f2 = np.asarray(freqs, dtype=np.float64) ** 2
    num = 12194.0**2 * f2**2
    den = (
        (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    )
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(num / den, 1e-20)) + 2.0


class Loudness(nn.Module):
    """A-weighted mean-square level per LogMel frame, in dB. A full-scale
    1 kHz sine reads about -3 dB; silence reads cfg.loudness_floor_db."""

    weights: torch.Tensor

    def __init__(self, cfg: FrontendConfig = FrontendConfig()):
        super().__init__()
        self.cfg = cfg
        self.stft = LogMel(cfg)
        freqs = np.linspace(0.0, cfg.sr / 2.0, cfg.n_fft // 2 + 1)
        w = 10.0 ** (a_weighting_db(freqs) / 10.0)
        # Parseval for a one-sided spectrum of a Hann-windowed frame: the
        # weighted power sum over bins, times 2 / (n_fft * sum(window^2)),
        # is the frame's mean square.
        win = torch.hann_window(cfg.win_length).double()
        scale = 2.0 / (cfg.n_fft * float((win**2).sum()))
        self.register_buffer(
            "weights", torch.from_numpy(w * scale).float(), persistent=False
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, N] -> [B, N // hop]
        power = self.stft.magnitude(audio) ** 2
        ms = torch.einsum("f,bft->bt", self.weights, power)
        db = 10.0 * torch.log10(ms.clamp(min=1e-12))
        return db.clamp(min=self.cfg.loudness_floor_db)
