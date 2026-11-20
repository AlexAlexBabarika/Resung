"""F0 with RMVPE, and resampling of F0 tracks between frame grids.

An F0 track here is Hz per frame, with 0 marking an unvoiced frame. RMVPE
frame i sits at time i * f0_hop / analysis_sr (10 ms steps). The vocoder
pairs mel frame j with the F0 at time j * hop / sr.
"""

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from config import FrontendConfig
from frontend.rmvpe import E2E, RMVPEMel

cfg = FrontendConfig()

F0_FRAME_S = cfg.f0_hop / cfg.analysis_sr  # 0.01


def f0_frames(n_analysis_samples: int) -> int:
    """RMVPE frames for a 16 kHz signal (centred STFT frames)."""
    return 1 + n_analysis_samples // cfg.f0_hop


def mel_frame_times(n_frames: int, start_frame: int = 0) -> np.ndarray:
    """Times, in seconds from the start of the signal, at which the vocoder
    expects the F0 of mel frames start_frame .. start_frame + n_frames - 1."""
    return (start_frame + np.arange(n_frames)) * cfg.hop_length / cfg.sr


def f0_at(f0: np.ndarray, times: np.ndarray, frame_s: float = F0_FRAME_S) -> np.ndarray:
    """Sample a track whose frame i sits at i * frame_s at arbitrary times.

    Voicing comes from the nearest frame. Between two voiced frames the pitch
    is interpolated on a log scale; next to an unvoiced frame the nearest
    voiced value is kept, so voicing edges don't smear into low pitches.
    Times past either end take the end frame.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    pos = np.clip(np.asarray(times, dtype=np.float64) / frame_s, 0.0, len(f0) - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(f0) - 1)
    w = pos - i0
    nearest = f0[np.rint(pos).astype(np.int64)]
    both = (f0[i0] > 0) & (f0[i1] > 0)
    lo = np.log2(np.where(both, f0[i0], 1.0))
    hi = np.log2(np.where(both, f0[i1], 1.0))
    out = np.where(both, 2.0 ** ((1.0 - w) * lo + w * hi), nearest)
    return np.where(nearest > 0, out, 0.0).astype(np.float32)


def interp_unvoiced(f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill unvoiced frames by log-scale interpolation between voiced ones
    (ends hold the nearest voiced value), as the vocoder's training data was.
    Returns (continuous F0, uv) with uv = 1 on frames that were unvoiced. An
    all-unvoiced track stays at 0."""
    f0 = np.asarray(f0, dtype=np.float64)
    uv = f0 <= 0
    out = f0.copy()
    if uv.any() and not uv.all():
        idx = np.arange(len(f0))
        out[uv] = 2.0 ** np.interp(idx[uv], idx[~uv], np.log2(f0[~uv]))
    return out.astype(np.float32), uv.astype(np.float32)


class RMVPE:
    """Batched RMVPE: 16 kHz audio [B, N] -> (F0 Hz, salience) [B, 1 + N // 160].
    F0 is 0 where the peak salience is at or below cfg.f0_threshold."""

    def __init__(self, device: str | torch.device = "cpu"):
        path = hf_hub_download(
            cfg.rmvpe_repo, cfg.rmvpe_file, revision=cfg.rmvpe_revision
        )
        self.device = torch.device(device)
        self.model = E2E(4, 1, (2, 2))
        self.model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        self.model.eval().to(self.device)
        self.mel = RMVPEMel().to(self.device)
        cents = 20.0 * torch.arange(360, dtype=torch.float32) + 1997.3794084376191
        self.cents = F.pad(cents, (4, 4)).to(self.device)  # zero cents on the pad

    @torch.no_grad()
    def __call__(self, audio16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio16 = audio16.to(self.device, torch.float32)
        mel = self.mel(audio16)  # [B, 128, T]
        n = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n))  # U-Net needs T % 32 == 0
        salience = self.model(mel)[:, :n]  # [B, T, 360]
        return self.decode(salience)

    def decode(self, salience: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Salience-weighted mean of the cents of the 9 bins around the peak.
        peak = salience.argmax(dim=-1, keepdim=True)  # [B, T, 1]
        padded = F.pad(salience, (4, 4))
        idx = peak + torch.arange(9, device=salience.device)  # [B, T, 9] in padded bins
        w = padded.gather(-1, idx)
        cents = (w * self.cents[idx]).sum(-1) / w.sum(-1).clamp(min=1e-8)
        conf = salience.amax(dim=-1)
        f0 = 10.0 * 2.0 ** (cents / 1200.0)
        f0 = torch.where(conf > cfg.f0_threshold, f0, torch.zeros_like(f0))
        return f0, conf
