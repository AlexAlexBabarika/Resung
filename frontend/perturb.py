"""The training-time pitch perturbation: a WORLD shift at 16 kHz.

Only the content encoder sees the result. F0, loudness and timbre always come
from the unshifted clip, so the decoder has to take its pitch from the F0
track, which is the situation inference puts it in.
"""

import numpy as np
import pyworld

from config import FrontendConfig

cfg = FrontendConfig()

FRAME_PERIOD_MS = 1000.0 * cfg.f0_hop / cfg.analysis_sr  # 10 ms, RMVPE's grid


def world_shift(
    audio16: np.ndarray, f0: np.ndarray, semitones: float, sr: int = cfg.analysis_sr
) -> np.ndarray:
    """Resynthesise `audio16` with WORLD at `semitones` from its pitch,
    keeping the spectral envelope (formants) where it is.

    f0: Hz per 10 ms frame with frame i at i * 10 ms, 0 when unvoiced,
    covering the clip (frontend.f0.f0_frames(len(audio16)) frames).
    """
    x = np.ascontiguousarray(audio16, dtype=np.float64)
    f0 = np.ascontiguousarray(f0, dtype=np.float64)
    t = np.arange(len(f0), dtype=np.float64) * FRAME_PERIOD_MS / 1000.0
    sp = pyworld.cheaptrick(x, f0, t, sr)
    ap = pyworld.d4c(x, f0, t, sr)
    y = pyworld.synthesize(f0 * 2.0 ** (semitones / 12.0), sp, ap, sr, FRAME_PERIOD_MS)
    n = len(x)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)
