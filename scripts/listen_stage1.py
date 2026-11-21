"""The step-4 check on a trained stage-1 checkpoint: is the unshifted
reconstruction close to transparent, and how do shifted outputs sound?

    uv run scripts/listen_stage1.py --ckpt best.pt [--cache-dir DIR] [--n 8]

On whole validation segments (held-out voices), with clean content as at
inference, it writes per clip:
    original         the segment
    vocoded          its own mel through the vocoder: the ceiling for Resung
    resung           the decoder's mel at the original pitch
    resung_shift+N   the decoder at F0 x 2^(N/12)
    pcnsf_shift+N    the vocoder's own shift of the original mel (baseline)
and prints, as medians over clips, how far the unshifted outputs' mels are
from the original and each shifted output's median pitch error.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FrontendConfig  # noqa: E402
from data import ResungDataset  # noqa: E402
from frontend.content import ContentVec  # noqa: E402
from frontend.mel import LogMel  # noqa: E402
from model import load_decoder  # noqa: E402
from vocoder import load_generator  # noqa: E402
from vocoder_check import median_pitch_cents  # noqa: E402

fe = FrontendConfig()
SHIFTS = (-6, -3, 3, 6)
SEG_FRAMES = round(fe.segment_seconds * fe.sr) // fe.hop_length  # 441


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--datasets", nargs="+", default=["vctk", "gtsinger"])
    p.add_argument("--out", type=Path, default=Path("listen_stage1"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    ds = ResungDataset(
        "val", dataset_keys=args.datasets, clip_frames=SEG_FRAMES, **kwargs
    )
    full = [i for i in range(len(ds)) if ds[i]["mask"].all()]
    picks = [
        full[j]
        for j in np.linspace(0, len(full) - 1, min(args.n, len(full))).astype(int)
    ]
    dev = torch.device(args.device)
    decoder = load_decoder(args.ckpt, dev)
    content_model = ContentVec().to(dev)
    gen = load_generator(dev)
    logmel = LogMel().to(dev)
    args.out.mkdir(parents=True, exist_ok=True)

    def render(mel, f0):
        return gen(mel, f0)[0, 0]

    rows = []
    with torch.no_grad():
        for i in picks:
            item = {k: v[None].to(dev) for k, v in ds[i].items()}
            name = Path(ds.files[i]).stem
            audio = item["audio"][0]
            mel = logmel(item["audio"])
            content = content_model(item["audio16"])
            track = item["f0"][0].cpu().numpy() * (1 - item["uv"][0].cpu().numpy())

            def decode(ratio: float):
                pred = decoder(
                    content,
                    item["f0"] * ratio,
                    item["uv"],
                    item["loudness"],
                    item["timbre"],
                    item["mask"],
                )
                return render(pred, item["f0"] * ratio)

            def mel_l1(y):
                return float((logmel(y[None]) - mel).abs()[..., 4:-4].mean())

            vocoded, resung = render(mel, item["f0"]), decode(1.0)
            row = {
                "clip": name,
                "vocoded_l1": mel_l1(vocoded),
                "resung_l1": mel_l1(resung),
            }
            outs = {"original": audio, "vocoded": vocoded, "resung": resung}
            for s in SHIFTS:
                r = 2.0 ** (s / 12.0)
                outs[f"resung_shift{s:+d}"] = decode(r)
                outs[f"pcnsf_shift{s:+d}"] = render(mel, item["f0"] * r)
                # Pitch error against the original track x ratio (10 ms grid).
                tgt = (
                    np.interp(
                        np.arange(0, len(audio) / fe.sr, 0.01),
                        np.arange(len(track)) * fe.hop_length / fe.sr,
                        track,
                    )
                    * r
                )
                for kind in ("resung", "pcnsf"):
                    y = outs[f"{kind}_shift{s:+d}"].cpu().numpy()
                    row[f"{kind}{s:+d}_cents"] = median_pitch_cents(y, tgt)
            for k, y in outs.items():
                sf.write(args.out / f"{name}_{k}.wav", y.cpu().numpy(), fe.sr)
            rows.append(row)
            print(
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )

    def median(key):
        return float(np.nanmedian([r[key] for r in rows]))

    print(
        f"\n{len(rows)} held-out clips, medians. Mel L1 to the original: vocoder "
        f"alone {median('vocoded_l1'):.3f}, Resung unshifted {median('resung_l1'):.3f} "
        "(transparent would be about the vocoder's figure)."
    )
    for s in SHIFTS:
        print(
            f"shift {s:+d}: pitch error Resung {median(f'resung{s:+d}_cents'):.1f} c, "
            f"vocoder's own shift {median(f'pcnsf{s:+d}_cents'):.1f} c"
        )
    print(f"wavs in {args.out}/")


if __name__ == "__main__":
    main()
