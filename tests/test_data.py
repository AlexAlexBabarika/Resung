"""ResungDataset on a synthetic cache (no models needed)."""

from pathlib import Path

import numpy as np
import pytest
import pyworld

from config import FrontendConfig
from data import ResungDataset, group_of
from frontend.audio import analysis_len, to_analysis_sr, to_int16
from frontend.f0 import f0_frames
from frontend.process import write_index
from tests.helpers import cents, tone

cfg = FrontendConfig()
HOP = cfg.hop_length
UNVOICED_HEAD = 5  # leading unvoiced F0 frames (50 ms) in every segment


def write_segment(path: Path, seconds: float, f0_hz: float, emb: np.ndarray) -> None:
    audio = tone(f0_hz, seconds, cfg.sr, harmonics=8, amp=0.3)
    nf = f0_frames(analysis_len(len(audio)))
    f0 = np.full(nf, f0_hz, np.float32)
    f0[:UNVOICED_HEAD] = 0.0
    np.savez(
        path,
        audio=to_int16(audio),
        src_sr=np.int32(48000),
        f0=f0,
        f0_conf=np.ones(nf, np.float16),
        loudness=np.full(len(audio) // HOP, -20.0, np.float16),
        timbre=emb.astype(np.float16),
    )


@pytest.fixture(scope="module")
def cache(tmp_path_factory) -> Path:
    # 60 VCTK-style speakers with 2 segments each (one full, one short),
    # plus a validation-only OpenSinger set.
    root = tmp_path_factory.mktemp("cache")
    rng = np.random.default_rng(0)
    (root / "vctk").mkdir()
    for s in range(60):
        f0 = 120.0 + 2.0 * s
        for j, seconds in enumerate([5.12, 2.0]):
            emb = rng.standard_normal(cfg.timbre_dim)
            write_segment(
                root / "vctk" / f"v_p{s:03d}__{j:06d}_0000.npz", seconds, f0, emb
            )
    (root / "opensinger_fem").mkdir()
    for s in range(10):
        emb = rng.standard_normal(cfg.timbre_dim)
        write_segment(
            root / "opensinger_fem" / f"o_f_{s}__000000_0000.npz", 5.12, 300.0, emb
        )
    write_index(root, "vctk")
    write_index(root, "opensinger_fem")
    return root


def make(cache: Path, split: str) -> ResungDataset:
    return ResungDataset(
        split, dataset_keys=["vctk", "opensinger_female"], cache_dir=cache
    )


def test_splits_hold_out_whole_voices_and_skip_validation_only_sets(cache):
    train, val = make(cache, "train"), make(cache, "val")
    train_groups = {group_of(f) for f in train.files}
    val_groups = {group_of(f) for f in val.files}
    assert train_groups and val_groups and not train_groups & val_groups
    assert not any("/opensinger_fem/" in f for f in train.files)  # share 0
    assert any("/opensinger_fem/" in f for f in val.files)
    assert len(train.weights) == len(train.files)


def test_item_shapes_and_alignment(cache):
    item = make(cache, "val")[0]
    n = cfg.clip_frames
    n16 = analysis_len(n * HOP)
    assert item["audio"].shape == (n * HOP,)
    assert item["audio16"].shape == item["audio16_pert"].shape == (n16,)
    for key in ("f0", "uv", "loudness", "mask"):
        assert item[key].shape == (n,), key
    assert item["timbre"].shape == (cfg.timbre_dim,)
    # Validation crops start at 0: the 50 ms unvoiced head covers the frames
    # the vocoder times before 0.045 s, and is filled with the voiced pitch.
    head = int(np.ceil(0.045 * cfg.sr / HOP))
    assert item["uv"][:head].tolist() == [1.0] * head
    assert item["uv"][head + 1 :].sum() == 0
    assert np.allclose(item["f0"].numpy(), item["f0"][-1].item())


def test_short_segments_are_padded_and_masked(cache):
    ds = make(cache, "val")
    i = next(i for i, f in enumerate(ds.files) if "__000001_" in f)  # the 2.0 s one
    item = ds[i]
    real = int(2.0 * cfg.sr) // HOP
    assert item["mask"].sum() == real
    assert item["audio"][int(2.0 * cfg.sr) :].abs().sum() == 0  # past the real samples
    assert (item["loudness"][real:] == cfg.loudness_floor_db).all()
    assert (item["uv"][real:] == 1).all()


def test_timbre_comes_from_another_segment_of_the_same_voice(cache):
    for split in ("train", "val"):
        ds = make(cache, split)
        for i in range(0, len(ds), 7):
            peers = ds.by_group[group_of(ds.files[i])]
            ref = ds._reference(i)
            if len(peers) > 1:
                assert ref != i and ref in peers


def test_the_perturbation_moves_only_the_content_input(cache):
    ds = make(cache, "val")
    item = ds[0]
    semis = float(item["perturb_st"])
    assert 0.5 <= abs(semis) <= cfg.perturb_st
    sr = cfg.analysis_sr
    f0_src = float(item["f0"][-1])

    def pitch(x) -> float:
        est, _ = pyworld.harvest(x.numpy().astype(np.float64), sr, frame_period=10.0)
        est = est[20:-20]
        return float(np.median(est[est > 0]))

    assert abs(cents(pitch(item["audio16"]), f0_src)) < 15
    assert abs(cents(pitch(item["audio16_pert"]), f0_src) - 100 * semis) < 15


def test_the_16k_crop_lines_up_with_the_target_crop(cache):
    ds = make(cache, "train")
    full = next(i for i, f in enumerate(ds.files) if "__000000_" in f)
    for _ in range(5):
        item = ds[full]
        ref = to_analysis_sr(item["audio"].numpy())
        got = item["audio16"].numpy()
        mid = slice(2000, len(got) - 2000)
        corr = np.corrcoef(ref[mid], got[mid])[0, 1]
        assert corr > 0.99
