import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from config import DataConfig
from frontend.process import Analyzer, process_and_save, write_index

DATASETS_DIR = Path("./datasets")


def _local_parquet_files(root: Path, pattern: str) -> list[str]:
    return sorted(str(p) for p in root.glob(pattern))


def _stream_parquet(files: list[str]):
    # Decoded at the native rate; process_and_save resamples with soxr.
    ds = load_dataset(
        "parquet", data_files={"train": files}, split="train", streaming=True
    )
    return ds.cast_column("audio", Audio())


def process_nsynth(
    parquet_root: Path,
    parquet_glob: str,
    out_dir: Path,
    budget_s: float,
    group_cap_s: float,
    analyzer: Analyzer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _local_parquet_files(parquet_root, parquet_glob)
    if not files:
        raise FileNotFoundError(f"No NSynth parquet files under {parquet_root}")

    ds = _stream_parquet(files)

    budget = {"remaining_s": budget_s}
    used: dict[str, float] = defaultdict(float)
    pbar = tqdm(total=int(budget_s), unit="s", desc="NSynth")
    # NSynth notes are 4s. Filter for vocal-range pitches (C2..C6).
    for i, ex in enumerate(ds):
        if budget["remaining_s"] <= 0:
            break
        pitch = ex.get("pitch", 60)
        inst = ex.get("instrument_str", f"note{i}")
        if pitch < 36 or pitch > 84 or used[inst] >= group_cap_s:
            continue
        wav = ex["audio"]["array"].astype(np.float32)
        src_sr = ex["audio"]["sampling_rate"]
        sec = process_and_save(
            wav, src_sr, out_dir, f"n_{inst}__{i:07d}", budget, analyzer
        )
        used[inst] += sec
        pbar.update(sec)
    pbar.close()


def process_vctk(
    parquet_root: Path,
    parquet_glob: str,
    out_dir: Path,
    budget_s: float,
    group_cap_s: float,
    analyzer: Analyzer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _local_parquet_files(parquet_root, parquet_glob)
    if not files:
        raise FileNotFoundError(f"No VCTK parquet files under {parquet_root}")

    ds = _stream_parquet(files)

    budget = {"remaining_s": budget_s}
    used: dict[str, float] = defaultdict(float)
    pbar = tqdm(total=int(budget_s), unit="s", desc="VCTK")
    for i, ex in enumerate(ds):
        if budget["remaining_s"] <= 0:
            break
        spk = ex.get("speaker_id", f"unk_{i}")
        # Every utterance is in the set twice, once per microphone.
        if used[spk] >= group_cap_s or "_mic2" in str(ex.get("file", "")):
            continue
        wav = ex["audio"]["array"].astype(np.float32)
        src_sr = ex["audio"]["sampling_rate"]
        sec = process_and_save(
            wav, src_sr, out_dir, f"v_{spk}__{i:06d}", budget, analyzer
        )
        used[spk] += sec
        pbar.update(sec)
    pbar.close()


def process_opensinger(
    parquet_root: Path,
    parquet_glob: str,
    out_dir: Path,
    budget_s: float,
    group_cap_s: float,
    analyzer: Analyzer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _local_parquet_files(parquet_root, parquet_glob)
    if not files:
        raise FileNotFoundError(f"No process_singer parquet files under {parquet_root}")

    ds = _stream_parquet(files)

    budget = {"remaining_s": budget_s}
    used: dict[str, float] = defaultdict(float)
    pbar = tqdm(total=int(budget_s), unit="s", desc="OpenSinger")
    for i, ex in enumerate(ds):
        if budget["remaining_s"] <= 0:
            break
        # OpenSinger ids are "<singer>_<song>_<n>"
        spk = str(ex.get("id", f"unk{i}")).split("_")[0]
        if used[spk] >= group_cap_s:
            continue
        wav = ex["audio"]["array"].astype(np.float32)
        src_sr = ex["audio"]["sampling_rate"]
        gender = ex.get("gender", "")
        sec = process_and_save(
            wav, src_sr, out_dir, f"o_{gender}_{spk}__{i:06d}", budget, analyzer
        )
        used[spk] += sec
        pbar.update(sec)
    pbar.close()


def process_gtsinger(
    wav_root: Path,
    wav_glob: str,
    out_dir: Path,
    budget_s: float,
    group_cap_s: float,
    analyzer: Analyzer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # <language>/<singer>/<technique>/<song>/<group>/NNNN.wav
    files = sorted(
        p for p in wav_root.glob(wav_glob) if "Paired_Speech_Group" not in p.parts
    )
    if not files:
        raise FileNotFoundError(f"No GTSinger wav files under {wav_root}")
    # Shuffled so each singer's cap is filled from a mix of songs and techniques.
    random.Random(0).shuffle(files)

    budget = {"remaining_s": budget_s}
    used: dict[str, float] = defaultdict(float)
    pbar = tqdm(total=int(budget_s), unit="s", desc="GTSinger")
    for i, path in enumerate(files):
        if budget["remaining_s"] <= 0:
            break
        singer = path.relative_to(wav_root).parts[1]
        if used[singer] >= group_cap_s:
            continue
        wav, src_sr = sf.read(path, dtype="float32", always_2d=True)
        sec = process_and_save(
            wav.mean(axis=1),
            src_sr,
            out_dir,
            f"g_{singer}__{i:06d}",
            budget,
            analyzer,
        )
        used[singer] += sec
        pbar.update(sec)
    pbar.close()


PROCESSORS = {
    "gtsinger": process_gtsinger,
    "opensinger_male": process_opensinger,
    "nsynth": process_nsynth,
    "vctk": process_vctk,
    "opensinger_female": process_opensinger,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Write Resung's 44.1 kHz training cache.")
    p.add_argument(
        "--datasets", nargs="+", help="subset of DataConfig.datasets_to_load"
    )
    p.add_argument(
        "--max-hours", type=float, help="cap every dataset's budget (smoke tests)"
    )
    p.add_argument("--cache-dir", type=Path, help="default: DataConfig.cache_dir")
    p.add_argument("--device", help="default: cuda, else mps, else cpu")
    args = p.parse_args()

    data_cfg = DataConfig()
    cache = args.cache_dir or Path(data_cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    analyzer = Analyzer(args.device)
    print(f"Processing datasets on {analyzer.device}...")
    for key in args.datasets or data_cfg.datasets_to_load:
        spec = data_cfg.datasets[key]
        budget_h = spec.length_hr
        if args.max_hours is not None:
            budget_h = min(budget_h, args.max_hours)
        PROCESSORS[key](
            DATASETS_DIR / spec.subdir,
            spec.parquet_glob,
            cache / spec.subdir,
            budget_h * 3600.0,
            spec.group_minutes * 60.0,
            analyzer,
        )
        write_index(cache, spec.subdir)


if __name__ == "__main__":
    main()
