"""Checks PC-NSF-HiFiGAN on cached clips before any decoder is trained.

    uv run scripts/vocoder_check.py [--cache-dir DIR] [--out DIR] [--n 12]

1. Copy synthesis: the clip's own mel and F0 through the vocoder. It should be
   close to transparent, and its output must line up with the input sample
   for sample (lag 0), or the waveform losses compare shifted signals.
2. The vocoder's own pitch shift: the same mel with F0 x 2^(s/12). This is the
   cheapest DSP-free baseline Resung's decoder has to beat.

Writes <out>/<clip>_{original,vocoded,shift+N}.wav for listening.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataConfig, FrontendConfig  # noqa: E402
from frontend.audio import from_int16, to_analysis_sr  # noqa: E402
from frontend.f0 import f0_at, interp_unvoiced, mel_frame_times  # noqa: E402
from frontend.mel import LogMel  # noqa: E402
from vocoder import load_generator  # noqa: E402

fe = FrontendConfig()
SHIFTS = (-6, -3, 3, 6)


def best_lag(y: np.ndarray, x: np.ndarray, max_lag: int = 1024, hop: int = 16) -> int:
    """Lag (samples, to `hop` resolution) that best aligns y to x; positive
    means y is late. Compares log-magnitude spectrograms, since the vocoder
    makes up its own harmonic phases and waveform correlation is meaningless."""

    def spec(a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(a.astype(np.float32))
        s = torch.stft(
            t, 1024, hop, window=torch.hann_window(1024), return_complex=True
        )
        return torch.log(s.abs().clamp(min=1e-5))

    sx, sy = spec(x), spec(y)
    m = max_lag // hop
    n = min(sx.shape[-1], sy.shape[-1]) - 2 * m
    ref = sx[:, m : m + n]
    errs = [
        float((sy[:, m + k : m + k + n] - ref).abs().mean()) for k in range(-m, m + 1)
    ]
    return (int(np.argmin(errs)) - m) * hop


def median_pitch_cents(y: np.ndarray, target_hz: np.ndarray) -> float:
    """Median deviation of y's F0 (harvest, 10 ms) from target_hz (10 ms grid)."""
    est, _ = pyworld.harvest(
        to_analysis_sr(y).astype(np.float64), fe.analysis_sr, 50, 1600, 10.0
    )
    n = min(len(est), len(target_hz))
    est, tgt = est[:n], target_hz[:n]
    both = (est > 0) & (tgt > 0)
    if both.sum() < 20:
        return float("nan")
    return float(np.median(np.abs(1200 * np.log2(est[both] / tgt[both]))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=Path(DataConfig().cache_dir))
    p.add_argument("--datasets", nargs="+", default=["vctk", "gtsinger"])
    p.add_argument("--out", type=Path, default=Path("vocoder_check"))
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    data = DataConfig()
    files = []
    for key in args.datasets:
        spec = data.datasets[key]
        index = json.loads((args.cache_dir / f"{spec.subdir}_index.json").read_text())
        random.Random(0).shuffle(index)
        files += index[: args.n // len(args.datasets)]
    args.out.mkdir(parents=True, exist_ok=True)
    gen = load_generator(args.device)
    logmel = LogMel().to(args.device)

    rows = []
    for f in files:
        with np.load(f) as d:
            audio = from_int16(d["audio"])
            track = d["f0"].astype(np.float32)
        audio = audio[: len(audio) // fe.hop_length * fe.hop_length]
        x = torch.from_numpy(audio)[None].to(args.device)
        mel = logmel(x)
        n = mel.shape[-1]
        f0_mel = f0_at(track, mel_frame_times(n))
        if (f0_mel > 0).mean() < 0.3:
            continue
        f0_cont, _ = interp_unvoiced(f0_mel)
        f0_t = torch.from_numpy(f0_cont)[None].to(args.device)
        with torch.no_grad():
            y = gen(mel, f0_t)[0, 0].cpu().numpy()
            mel_y = logmel(torch.from_numpy(y)[None].to(args.device))
        lag = best_lag(y, audio)
        mel_l1 = float((mel_y - mel).abs()[..., 4:-4].mean())
        row = {"clip": Path(f).stem, "lag": lag, "mel_l1": mel_l1}
        name = Path(f).stem
        sf.write(args.out / f"{name}_original.wav", audio, fe.sr)
        sf.write(args.out / f"{name}_vocoded.wav", y, fe.sr)
        for s in SHIFTS:
            ratio = 2.0 ** (s / 12.0)
            with torch.no_grad():
                ys = gen(mel, f0_t * ratio)[0, 0].cpu().numpy()
            sf.write(args.out / f"{name}_shift{s:+d}.wav", ys, fe.sr)
            row[f"cents{s:+d}"] = median_pitch_cents(ys, track * ratio)
        rows.append(row)
        print(row)

    lags = [r["lag"] for r in rows]
    print(f"\n{len(rows)} clips. Alignment lag (samples): {sorted(set(lags))}")
    print(f"copy-synthesis log-mel L1: mean {np.mean([r['mel_l1'] for r in rows]):.3f}")
    for s in SHIFTS:
        v = [r[f"cents{s:+d}"] for r in rows if np.isfinite(r[f"cents{s:+d}"])]
        print(
            f"vocoder's own shift {s:+d}: median pitch error {np.median(v):.1f} cents"
        )
    print(f"wavs in {args.out}/")


if __name__ == "__main__":
    main()
