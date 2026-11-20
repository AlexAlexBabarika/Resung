"""The cache writer, end to end on a synthetic recording (needs the models)."""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from config import FrontendConfig
from frontend.audio import analysis_len, from_int16, to_analysis_sr
from frontend.f0 import f0_frames
from frontend.process import SEG_LEN, Analyzer, process_and_save, write_index
from tests.helpers import cents, tone

cfg = FrontendConfig()

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer("cpu")


def test_recording_is_segmented_normalised_and_analysed(tmp_path, analyzer):
    # 7 s at 48 kHz: one full 5.12 s segment and a 1.88 s remainder (kept,
    # since it is longer than 1.28 s).
    audio = tone(220.0, 7.0, 48000, harmonics=10, amp=0.05)
    out = tmp_path / "vctk"
    out.mkdir()
    budget = {"remaining_s": 100.0}
    written = process_and_save(audio, 48000, out, "v_p999__000001", budget, analyzer)
    files = sorted(out.glob("*.npz"))
    assert [f.name for f in files] == [
        "v_p999__000001_0000.npz",
        "v_p999__000001_0001.npz",
    ]
    assert np.isclose(written, 7.0, atol=0.05)
    assert np.isclose(budget["remaining_s"], 100.0 - written)

    with np.load(files[0]) as d:
        assert len(d["audio"]) == SEG_LEN and d["audio"].dtype == np.int16
        assert int(d["src_sr"]) == 48000
        assert (
            d["f0"].shape == d["f0_conf"].shape == (f0_frames(analysis_len(SEG_LEN)),)
        )
        assert d["loudness"].shape == (SEG_LEN // cfg.hop_length,)
        assert d["timbre"].shape == (cfg.timbre_dim,)
        voiced = d["f0"][d["f0"] > 0]
        assert len(voiced) > 0.9 * len(d["f0"])
        assert abs(cents(float(np.median(voiced)), 220.0)) < 20
        # Loudness-normalised up from a quiet input: about -23 LUFS.
        assert float(np.median(d["loudness"])) > -30.0

    with np.load(files[1]) as d:
        n = len(d["audio"])
        assert n < SEG_LEN
        assert d["f0"].shape == (f0_frames(analysis_len(n)),)
        assert d["loudness"].shape == (n // cfg.hop_length,)
        # The short segment's timbre is taken from its real samples only.
        x16 = torch.from_numpy(to_analysis_sr(from_int16(d["audio"])))[None]
        direct = analyzer.timbre(x16)
        stored = torch.from_numpy(d["timbre"].astype(np.float32))[None]
        assert F.cosine_similarity(direct, stored).item() > 0.999

    write_index(tmp_path, "vctk")
    index = json.loads((tmp_path / "vctk_index.json").read_text())
    emb = np.load(tmp_path / "vctk_timbre.npy")
    assert index == [str(f) for f in files] and emb.shape == (2, cfg.timbre_dim)


def test_budget_stops_mid_recording(tmp_path, analyzer):
    audio = tone(220.0, 12.0, 44100, harmonics=10, amp=0.05)
    budget = {"remaining_s": 3.0}
    written = process_and_save(audio, 44100, tmp_path, "g_s1__000001", budget, analyzer)
    assert len(list(tmp_path.glob("*.npz"))) == 1  # one 5.12 s segment covers 3 s
    assert np.isclose(written, 5.12, atol=0.01) and budget["remaining_s"] < 0
