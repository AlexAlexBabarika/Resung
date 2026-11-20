import librosa
import numpy as np
import pytest
import pyworld
import torch
import torch.nn.functional as F

from config import FrontendConfig
from frontend.audio import analysis_len, from_int16, resample, segments, to_int16
from frontend.f0 import f0_at, f0_frames, interp_unvoiced, mel_frame_times
from frontend.mel import LogMel, Loudness
from frontend.perturb import world_shift
from tests.helpers import cents, tone

cfg = FrontendConfig()


def test_log_mel_frame_count_and_peak_band():
    x = torch.from_numpy(tone(440.0, 1.0, cfg.sr))[None]
    m = LogMel()(x)
    assert m.shape == (1, cfg.n_mels, x.shape[-1] // cfg.hop_length)
    centres = librosa.mel_frequencies(cfg.n_mels + 2, fmin=cfg.fmin, fmax=cfg.fmax)[
        1:-1
    ]
    peak = int(m[0, :, 5:-5].mean(-1).argmax())
    assert abs(peak - int(np.abs(centres - 440.0).argmin())) <= 1


def test_loudness_reference_levels():
    loud = Loudness()

    def level(x: np.ndarray) -> float:
        return float(loud(torch.from_numpy(x)[None])[0, 5:-5].mean())

    full = tone(1000.0, 1.0, cfg.sr, amp=1.0)
    assert abs(level(full) - (-3.01)) < 0.3  # mean square 0.5
    assert abs(level(full) - level(0.1 * full) - 20.0) < 0.1
    # A-weighting puts 100 Hz about 19.1 dB below 1 kHz.
    assert abs(level(full) - level(tone(100.0, 1.0, cfg.sr, amp=1.0)) - 19.1) < 0.6
    silence = loud(torch.zeros(1, cfg.sr))
    assert silence.shape == (1, cfg.sr // cfg.hop_length)
    assert torch.all(silence == cfg.loudness_floor_db)


def test_f0_at_interpolates_on_a_log_scale_and_keeps_voicing():
    track = np.array([100, 200, 0, 0, 400], np.float32)
    times = np.array([0.0, 0.005, 0.01, 0.014, 0.016, 0.02, 0.04, 0.09])
    out = f0_at(track, times, frame_s=0.01)
    assert out[0] == 100
    assert np.isclose(out[1], np.sqrt(100 * 200), atol=0.01)  # halfway in log
    assert out[2] == 200
    assert out[3] == 200  # nearest frame voiced, next one not: no smearing
    assert out[4] == 0 and out[5] == 0
    assert out[6] == 400 and out[7] == 400  # past the end holds the last frame


def test_interp_unvoiced_fills_gaps_and_flags_them():
    f0, uv = interp_unvoiced(np.array([0, 100, 0, 400, 0], np.float32))
    assert np.allclose(f0, [100, 100, 200, 400, 400])
    assert uv.tolist() == [1, 0, 1, 0, 1]
    f0, uv = interp_unvoiced(np.zeros(3, np.float32))
    assert (f0 == 0).all() and (uv == 1).all()


def test_mel_frame_times_follow_the_vocoder_convention():
    assert np.allclose(
        mel_frame_times(3, start_frame=2), np.array([2, 3, 4]) * 512 / 44100
    )


@pytest.mark.parametrize(
    "n", [cfg.clip_frames * cfg.hop_length, round(5.12 * cfg.sr), 12345]
)
def test_analysis_len_matches_the_resampler(n):
    assert len(
        resample(np.zeros(n, np.float32), cfg.sr, cfg.analysis_sr)
    ) == analysis_len(n)


def test_segments_pad_the_last_piece_and_drop_short_ones():
    out = list(segments(np.ones(25, np.float32), 10, 6))
    assert [n for _, n in out] == [10, 10]
    out = list(segments(np.ones(27, np.float32), 10, 6))
    assert [n for _, n in out] == [10, 10, 7]
    assert out[-1][0].shape == (10,) and out[-1][0][7:].sum() == 0


def test_int16_round_trip():
    x = np.linspace(-1, 1, 101, dtype=np.float32)
    assert np.abs(from_int16(to_int16(x)) - x).max() < 1e-4


def test_world_shift_moves_the_pitch_and_keeps_the_length():
    sr = cfg.analysis_sr
    x = tone(200.0, 1.0, sr, harmonics=8, amp=0.5)
    f0 = np.full(f0_frames(len(x)), 200.0, np.float32)
    for semis, target in [(12.0, 400.0), (-5.0, 200.0 * 2 ** (-5 / 12))]:
        y = world_shift(x, f0, semis)
        assert y.shape == x.shape
        est, _ = pyworld.harvest(y.astype(np.float64), sr, frame_period=10.0)
        est = est[20:-20]
        assert abs(cents(float(np.median(est[est > 0])), target)) < 15


@pytest.fixture(scope="module")
def rmvpe():
    from frontend.f0 import RMVPE

    return RMVPE("cpu")


@pytest.mark.models
def test_rmvpe_tracks_a_harmonic_tone(rmvpe):
    sr = cfg.analysis_sr
    x = tone(220.0, 2.0, sr, harmonics=10, amp=0.3)
    f0, conf = rmvpe(torch.from_numpy(x)[None])
    assert f0.shape == conf.shape == (1, f0_frames(len(x)))
    mid = f0[0, 10:-10]
    assert (mid > 0).float().mean() > 0.9
    assert abs(cents(float(mid[mid > 0].median()), 220.0)) < 20
    silent, _ = rmvpe(torch.zeros(1, sr))
    assert (silent == 0).all()


@pytest.mark.models
def test_contentvec_loads_every_weight():
    from frontend.content import ContentVec, content_frames

    cv = ContentVec()  # raises if any checkpoint tensor went unused or missing
    cv.train()
    assert not cv.model.training
    y = cv(0.1 * torch.randn(2, 16000))
    assert y.shape == (2, content_frames(16000), 768)


@pytest.mark.models
def test_timbre_embedding_is_deterministic_across_a_batch():
    from frontend.timbre import Timbre

    timbre = Timbre("cpu")
    x = torch.from_numpy(tone(150.0, 2.0, 16000, harmonics=20, amp=0.3))[None]
    e = timbre(torch.cat([x, x]))
    assert e.shape == (2, cfg.timbre_dim)
    assert torch.allclose(e[0], e[1], atol=1e-4)
    alone = timbre(x)
    assert F.cosine_similarity(alone, e[:1]).item() > 0.9999
