"""Writes the training cache: one .npz per segment of up to 5.12 s.

Everything that doesn't depend on the training-time perturbation is computed
here once: the 44.1 kHz audio, RMVPE F0, loudness and the timbre embedding.
Content features are computed on the GPU during training, from the
perturbed audio.

Each <id>.npz holds:
    audio     int16 [real_len] at 44.1 kHz, loudness-normalised
    src_sr    the source's sample rate (16 kHz OpenSinger has no top octave)
    f0        float32 [f0_frames], Hz on RMVPE's 10 ms grid, 0 = unvoiced
    f0_conf   float16 [f0_frames], RMVPE peak salience
    loudness  float16 [real_len // 512], dB, on the vocoder's mel frames
    timbre    float16 [192], ECAPA embedding of the segment

Segmentation, trimming and loudness normalisation follow the old cache, so
each speaker or singer ends up with the same number of segments and the
speaker-held-out split comes out the same.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import torch

from config import FrontendConfig
from frontend.audio import (
    analysis_len,
    normalization_gain,
    resample,
    segments,
    to_analysis_sr,
    to_int16,
)
from frontend.f0 import RMVPE, f0_frames
from frontend.mel import Loudness
from frontend.timbre import Timbre

cfg = FrontendConfig()

SEG_LEN = round(cfg.segment_seconds * cfg.sr)
# As in the old cache: drop pieces shorter than half a 2.56 s clip.
MIN_SEG_LEN = round(1.28 * cfg.sr)
# The old cache trimmed at 22.05 kHz with 2048-sample frames and a 512 hop;
# the same durations at 44.1 kHz.
TRIM_FRAME, TRIM_HOP = 4096, 1024


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Analyzer:
    """The models that fill the cache, loaded once."""

    def __init__(self, device: str | None = None):
        self.device = device or default_device()
        self.rmvpe = RMVPE(self.device)
        self.timbre = Timbre(self.device)
        self.loudness = Loudness().to(self.device)

    @torch.no_grad()
    def analyse(self, clips: np.ndarray, real_lens: list[int]) -> list[dict]:
        # clips: [B, SEG_LEN] at cfg.sr, zero-padded past each real_len
        x16 = np.stack([to_analysis_sr(c) for c in clips])
        x16_t = torch.from_numpy(x16)
        f0, conf = self.rmvpe(x16_t)
        # One at a time on the real samples: padding shifts ECAPA embeddings.
        emb = [
            self.timbre(x16_t[i : i + 1, : analysis_len(n)])[0]
            for i, n in enumerate(real_lens)
        ]
        loud = self.loudness(torch.from_numpy(clips).to(self.device))
        out = []
        for i, n in enumerate(real_lens):
            nf = f0_frames(analysis_len(n))
            out.append(
                {
                    "f0": f0[i, :nf].cpu().numpy().astype(np.float32),
                    "f0_conf": conf[i, :nf].cpu().numpy().astype(np.float16),
                    "loudness": loud[i, : n // cfg.hop_length]
                    .cpu()
                    .numpy()
                    .astype(np.float16),
                    "timbre": emb[i].cpu().numpy().astype(np.float16),
                }
            )
        return out


def process_and_save(
    audio: np.ndarray,
    src_sr: int,
    out_dir: Path,
    base_id: str,
    budget: dict,
    analyzer: Analyzer,
) -> float:
    """Resample, trim, normalise and segment one recording, analyse the
    segments and write them as <base_id>_NNNN.npz. Spends seconds from
    budget["remaining_s"] and returns the seconds written."""
    if budget["remaining_s"] <= 0:
        return 0.0
    audio = resample(audio, src_sr, cfg.sr)
    audio, _ = librosa.effects.trim(
        audio, top_db=30, frame_length=TRIM_FRAME, hop_length=TRIM_HOP
    )
    if len(audio) < cfg.sr * 0.5:
        return 0.0
    audio = (audio * normalization_gain(audio)).astype(np.float32)

    clips: list[np.ndarray] = []
    real_lens: list[int] = []
    remaining = budget["remaining_s"]
    for clip, real_len in segments(audio, SEG_LEN, MIN_SEG_LEN):
        if remaining <= 0:
            break
        clips.append(clip)
        real_lens.append(real_len)
        remaining -= real_len / cfg.sr
    if not clips:
        return 0.0

    feats = analyzer.analyse(np.stack(clips), real_lens)
    written = 0.0
    for i, (clip, real_len, feat) in enumerate(zip(clips, real_lens, feats)):
        np.savez(
            out_dir / f"{base_id}_{i:04d}.npz",
            audio=to_int16(clip[:real_len]),
            src_sr=np.int32(src_sr),
            **feat,
        )
        seconds = real_len / cfg.sr
        written += seconds
        budget["remaining_s"] -= seconds
    return written


def write_index(cache: Path, subdir: str) -> None:
    """<subdir>_index.json lists the segments; <subdir>_timbre.npy holds
    their timbre embeddings in the same order, so the loader can draw a
    reference from another segment without opening it."""
    files = sorted(str(p) for p in (cache / subdir).glob("*.npz"))
    (cache / f"{subdir}_index.json").write_text(json.dumps(files))
    emb = np.zeros((len(files), cfg.timbre_dim), dtype=np.float16)
    for i, f in enumerate(files):
        with np.load(f) as d:
            emb[i] = d["timbre"]
    np.save(cache / f"{subdir}_timbre.npy", emb)
