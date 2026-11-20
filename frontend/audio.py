"""Waveform utilities shared by data processing, the loader and inference."""

import math
from collections.abc import Iterator

import numpy as np
import pyloudnorm
import soxr

from config import FrontendConfig

cfg = FrontendConfig()

_METERS: dict[int, pyloudnorm.Meter] = {}


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """soxr HQ resampling. Every 44.1 kHz -> 16 kHz conversion goes through
    here, so the analysis models see the same signal in training and at
    inference."""
    audio = np.asarray(audio, dtype=np.float32)
    if sr_in == sr_out:
        return audio
    return soxr.resample(audio, sr_in, sr_out, quality="HQ").astype(np.float32)


def to_analysis_sr(audio: np.ndarray) -> np.ndarray:
    return resample(audio, cfg.sr, cfg.analysis_sr)


def analysis_len(n_samples: int) -> int:
    """Length of `n_samples` at cfg.sr once resampled to cfg.analysis_sr."""
    return round(n_samples * cfg.analysis_sr / cfg.sr)


def normalization_gain(audio: np.ndarray, sr: int = cfg.sr) -> float:
    # Integrated loudness to cfg.target_lufs, then a peak ceiling of 0.99.
    # Training data and inference input both go through this.
    meter = _METERS.setdefault(sr, pyloudnorm.Meter(rate=sr))
    gain = 1.0
    try:
        loud = meter.integrated_loudness(audio)
        if math.isfinite(loud):
            gain = 10.0 ** ((cfg.target_lufs - loud) / 20.0)
    except ValueError:  # shorter than one 400 ms gating block
        pass
    peak = float(np.max(np.abs(audio))) * gain + 1e-9
    if peak > 0.99:
        gain *= 0.99 / peak
    return gain


def segments(
    audio: np.ndarray, seg_len: int, min_len: int
) -> Iterator[tuple[np.ndarray, int]]:
    """Consecutive seg_len pieces, the last zero-padded; pieces with fewer
    than min_len real samples are dropped. Yields (padded piece, real length)."""
    for start in range(0, len(audio), seg_len):
        piece = audio[start : start + seg_len]
        real_len = len(piece)
        if real_len < min_len:
            continue
        if real_len < seg_len:
            piece = np.pad(piece, (0, seg_len - real_len))
        yield piece, real_len


def to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)


def from_int16(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32) / 32767.0
