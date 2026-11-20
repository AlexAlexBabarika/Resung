"""Sanity-check the training cache before paying for GPU hours.

    uv run scripts/check_cache.py [--cache-dir DIR] [--datasets KEY ...]

Fails loudly on the mistakes the old pipeline made: a constant F0 (CREPE
once returned 2 kHz everywhere), a single voice (no validation split), and
now also a band-limited source used as a 44.1 kHz training target.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataConfig, FrontendConfig  # noqa: E402

FULL_BAND_SR = 44100


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--datasets", nargs="+")
    args = p.parse_args()
    data, fe = DataConfig(), FrontendConfig()
    cache = args.cache_dir or Path(data.cache_dir)
    random.seed(0)

    for key in args.datasets or data.datasets_to_load:
        spec = data.datasets[key]
        files = json.loads((cache / f"{spec.subdir}_index.json").read_text())
        emb = np.load(cache / f"{spec.subdir}_timbre.npy").astype(np.float32)
        assert len(files) == len(emb), f"{key}: index and timbre table differ"
        assert np.isfinite(emb).all() and (np.linalg.norm(emb, axis=1) > 0).all(), (
            f"{key}: bad timbre embeddings"
        )
        groups = {Path(f).name.split("__")[0] for f in files}
        voiced, f0, loud, seconds, src_srs = [], [], [], 0.0, set()
        for f in random.sample(files, min(200, len(files))):
            with np.load(f) as d:
                assert d["audio"].dtype == np.int16, f"{f}: audio is not int16"
                v = d["f0"] > 0
                voiced.append(v.mean())
                f0.extend(d["f0"][v].tolist())
                loud.append(float(np.percentile(d["loudness"], 90)))
                seconds += len(d["audio"]) / fe.sr
                src_srs.add(int(d["src_sr"]))
        hours = seconds / min(200, len(files)) * len(files) / 3600
        med = float(np.median(f0)) if f0 else float("nan")
        role = "train + val" if spec.share > 0 else "val only"
        print(
            f"{key} ({role}): {len(files)} segments, ~{hours:.1f} h, {len(groups)} voices, "
            f"{100 * np.mean(voiced):.0f}% frames voiced, median F0 {med:.0f} Hz, "
            f"loud frames (p90) {np.median(loud):.0f} dB, source rates {sorted(src_srs)}"
        )
        assert len(groups) > 1, f"{key}: need at least 2 voices for a validation split"
        assert 0.15 < np.mean(voiced) < 0.98, f"{key}: voiced share looks wrong"
        assert 70 < med < 900, f"{key}: median F0 looks wrong"
        assert -35 < np.median(loud) < -5, f"{key}: loudness normalisation looks wrong"
        if spec.share > 0:
            assert min(src_srs) >= FULL_BAND_SR, (
                f"{key}: sources at {sorted(src_srs)} Hz lack the top octave; "
                "a 44.1 kHz target needs full-band audio (set share=0 to keep it for val)"
            )
    print("cache looks sane")


if __name__ == "__main__":
    main()
