"""Training and validation clips for Resung.

Each item is one crop of a cached segment (frontend/process.py):

    audio         [clip_frames * hop]  44.1 kHz target y (the unshifted clip)
    audio16       [n16]                the same crop at 16 kHz, clean
    audio16_pert  [n16]                that crop WORLD-shifted by perturb_st,
                                       for the content encoder only
    perturb_st    []                   the perturbation in semitones
    f0            [clip_frames]        Hz on the vocoder's frame times, with
                                       unvoiced frames interpolated
    uv            [clip_frames]        1 where the frame is unvoiced
    loudness      [clip_frames]        dB
    timbre        [192]                ECAPA embedding of another segment by
                                       the same speaker or singer
    mask          [clip_frames]        1 on frames that hold real audio

F0, loudness and timbre describe the unshifted clip; only the content input
is perturbed. The decoder therefore has to take its pitch from the F0 track.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import DataConfig, FrontendConfig, TrainConfig
from frontend.audio import analysis_len, from_int16, to_analysis_sr
from frontend.f0 import F0_FRAME_S, f0_at, f0_frames, interp_unvoiced, mel_frame_times
from frontend.perturb import world_shift

data_cfg = DataConfig()
fe_cfg = FrontendConfig()
train_cfg = TrainConfig()


def _split_by_group(files: list[str], val_split: float):
    # Unchanged from the old pipeline, so the held-out speakers and singers
    # are the ones eval_final.json was measured on. Clip names are
    # "<group>__<rest>.npz", group being the speaker or singer. Whole groups
    # go to val so no voice is on both sides.
    groups: dict[str, list[str]] = {}
    for f in files:
        name = Path(f).name
        assert "__" in name, f"{name} has no group prefix; rerun Process_HF_Datasets.py"
        groups.setdefault(name.split("__")[0], []).append(f)
    assert len(groups) > 1, f"need at least 2 groups to split, got {list(groups)}"
    keys = sorted(groups)
    random.Random(0).shuffle(keys)
    val: list[str] = []
    while len(val) < len(files) * val_split and len(keys) > 1:
        val.extend(groups[keys.pop()])
    return [f for k in keys for f in groups[k]], val


def group_of(path: str) -> str:
    return Path(path).name.split("__")[0]


class ResungDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        dataset_keys: list[str] | None = None,
        cache_dir: str | Path | None = None,
        clip_frames: int = fe_cfg.clip_frames,
        perturb_st: float = fe_cfg.perturb_st,
    ):
        cache = Path(cache_dir or data_cfg.cache_dir)
        self.is_train = split == "train"
        self.clip_frames = clip_frames
        self.perturb_st = perturb_st
        self.files: list[str] = []
        # share / (segments of that dataset in this split): sampling with these
        # weights draws each dataset in proportion to its share, whatever its
        # size. Datasets with share 0 are validation-only.
        self.weights: list[float] = []
        timbres: list[np.ndarray] = []
        for key in dataset_keys or data_cfg.datasets_to_load:
            spec = data_cfg.datasets[key]
            if self.is_train and spec.share <= 0:
                continue
            index = json.loads((cache / f"{spec.subdir}_index.json").read_text())
            emb = np.load(cache / f"{spec.subdir}_timbre.npy")
            assert len(emb) == len(index), f"{spec.subdir}: rerun write_index"
            row = {f: i for i, f in enumerate(index)}
            train, val = _split_by_group(index, data_cfg.val_split)
            part = train if self.is_train else val
            self.files.extend(part)
            self.weights.extend([spec.share / max(1, len(part))] * len(part))
            timbres.append(emb[[row[f] for f in part]])
        self.timbre = (
            np.concatenate(timbres).astype(np.float32)
            if timbres
            else np.zeros((0, fe_cfg.timbre_dim), np.float32)
        )
        self.by_group: dict[str, list[int]] = {}
        for i, f in enumerate(self.files):
            self.by_group.setdefault(group_of(f), []).append(i)

    def __len__(self) -> int:
        return len(self.files)

    def _reference(self, index: int) -> int:
        """Another segment by the same voice, so the timbre input can't carry
        this clip's pitch or words. Validation always takes the next one."""
        peers = self.by_group[group_of(self.files[index])]
        if len(peers) == 1:
            return index
        if self.is_train:
            j = random.randrange(len(peers) - 1)
            j = j if peers[j] != index else len(peers) - 1
            return peers[j]
        return peers[(peers.index(index) + 1) % len(peers)]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.files[index]) as d:
            audio = from_int16(d["audio"])
            f0_track = d["f0"].astype(np.float32)
            loud_track = d["loudness"].astype(np.float32)

        hop = fe_cfg.hop_length
        n_frames = self.clip_frames
        clip_len = n_frames * hop
        max_k = max(0, (len(audio) - clip_len) // hop)
        k = random.randint(0, max_k) if self.is_train else 0
        if self.is_train:
            semis = random.uniform(-self.perturb_st, self.perturb_st)
        else:
            rng = random.Random(index)
            semis = rng.uniform(-self.perturb_st, self.perturb_st)
            if abs(semis) < 0.5:
                semis = 0.5 if semis >= 0 else -0.5

        # 44.1 kHz target, zero-padded past the end of the segment.
        crop = audio[k * hop : k * hop + clip_len]
        real_frames = len(crop) // hop
        crop = np.pad(crop, (0, clip_len - len(crop)))
        mask = np.zeros(n_frames, np.float32)
        mask[:real_frames] = 1.0

        # 16 kHz crop, cut from the resampled segment so it has real context
        # at both edges. Starts at the same instant as the 44.1 kHz crop.
        seg16 = to_analysis_sr(audio)
        start16 = analysis_len(k * hop)
        n16 = analysis_len(clip_len)
        crop16 = seg16[start16 : start16 + n16]
        crop16 = np.pad(crop16, (0, n16 - len(crop16)))

        # F0 on the vocoder's frame times, and on WORLD's 10 ms grid for the crop.
        f0_mel = f0_at(f0_track, mel_frame_times(n_frames, start_frame=k))
        f0_mel[mask == 0] = 0.0
        f0_cont, uv = interp_unvoiced(f0_mel)
        f0_world = f0_at(
            f0_track,
            start16 / fe_cfg.analysis_sr + np.arange(f0_frames(n16)) * F0_FRAME_S,
        )
        pert16 = world_shift(crop16, f0_world, semis)

        loud = np.full(n_frames, fe_cfg.loudness_floor_db, np.float32)
        seg_loud = loud_track[k : k + n_frames]
        loud[: len(seg_loud)] = seg_loud

        return {
            "audio": torch.from_numpy(crop),
            "audio16": torch.from_numpy(crop16),
            "audio16_pert": torch.from_numpy(pert16),
            "perturb_st": torch.tensor(semis, dtype=torch.float32),
            "f0": torch.from_numpy(f0_cont),
            "uv": torch.from_numpy(uv),
            "loudness": torch.from_numpy(loud),
            "timbre": torch.from_numpy(self.timbre[self._reference(index)]),
            "mask": torch.from_numpy(mask),
        }

    @staticmethod
    def make_loader(split="train", batch_size=None, num_workers=None, **kwargs):
        ds = ResungDataset(split, **kwargs)
        is_train = split == "train"
        if num_workers is None:
            num_workers = train_cfg.num_workers
        nw = num_workers if is_train else min(4, num_workers)
        sampler = None
        if is_train:
            sampler = WeightedRandomSampler(ds.weights, len(ds), replacement=True)
        return DataLoader(
            ds,
            batch_size=batch_size or train_cfg.batch_size,
            sampler=sampler,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train,
            persistent_workers=(nw > 0),
            multiprocessing_context="spawn" if nw > 0 else None,
        )
