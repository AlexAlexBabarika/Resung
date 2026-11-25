import numpy as np
import pytest
import torch

from speaker_similarity import SpeakerSimilarity, pieces

SR = 16000


def test_pieces_cover_the_clip_and_drop_a_short_tail():
    assert pieces(50, 20, 5) == [(0, 20), (20, 40), (40, 50)]
    assert pieces(43, 20, 5) == [(0, 20), (20, 40)]  # 3-sample tail dropped
    assert pieces(3, 20, 5) == [(0, 3)]  # the only piece stays, however short
    assert pieces(40, 20, 5) == [(0, 20), (20, 40)]


def voice(f0: float, formant: float, seconds: float, seed: int) -> np.ndarray:
    """A crude sung vowel: harmonics of a slowly wobbling f0 under one
    resonance at `formant` Hz, plus a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))) / SR
    x = np.zeros_like(t)
    for h in range(1, int(7000 // f0)):
        x += np.sin(h * phase) / (1 + ((h * f0 - formant) / 300.0) ** 2)
    x += 0.01 * rng.standard_normal(len(t))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


@pytest.fixture(scope="module")
def score():
    return SpeakerSimilarity("cpu")


@pytest.mark.models
def test_a_clip_matches_itself_at_any_level(score):
    x = voice(150.0, 700.0, 3.0, seed=0)
    e = score.embed(x)
    assert e.shape == (512,) and torch.isclose(e.norm(), torch.tensor(1.0), atol=1e-5)
    assert score(x, x) > 0.9999
    assert score(x, 0.1 * x) > 0.999


@pytest.mark.models
def test_a_different_voice_scores_below_the_same_voice(score):
    a1 = voice(150.0, 700.0, 3.0, seed=0)
    a2 = voice(150.0, 700.0, 3.0, seed=1)  # same "voice", other noise
    b = voice(260.0, 1400.0, 3.0, seed=2)
    assert score(a1, a2) > score(a1, b) + 0.05


@pytest.mark.models
def test_long_audio_is_embedded_in_pieces(score):
    x = voice(150.0, 700.0, 45.0, seed=0)  # 20 s + 20 s + 5 s
    assert score(x, x[: 20 * SR]) > 0.98


def test_pick_windows_takes_the_two_most_sung_stretches_without_overlap():
    from scripts.voice_check import pick_windows

    voiced = np.zeros(100, bool)
    voiced[10:30] = True  # 20 voiced frames
    voiced[60:75] = True  # 15
    a, b = pick_windows(voiced, 20)
    assert a == 10 and 41 <= b <= 60 and voiced[b : b + 20].sum() == 15
    # One long sung stretch: the second window sits next to the first.
    a, b = pick_windows(np.ones(50, bool), 20)
    assert abs(a - b) >= 20
    with pytest.raises(ValueError):
        pick_windows(np.ones(39, bool), 20)


def test_loudest_finds_the_stretch_with_the_most_energy():
    from scripts.voice_check import loudest

    x = np.zeros((2, 100), np.float32)
    x[:, 40:60] = 1.0
    x[:, 70:80] = 0.5
    out = loudest(x, 20, step=10)
    assert out.shape == (2, 20) and (out == 1.0).all()
    with pytest.raises(ValueError):
        loudest(x, 101)
