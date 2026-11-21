import pytest
import torch

from losses import MultiResSTFTLoss, WaveLoss, masked_l1

HOP = 512


def test_masked_l1_ignores_padded_frames():
    pred, tgt = torch.zeros(1, 4, 10), torch.zeros(1, 4, 10)
    pred[..., 6:] = 100.0
    mask = torch.ones(1, 10)
    mask[:, 6:] = 0
    assert masked_l1(pred, tgt, mask) == 0
    pred[..., :6] = 1.0
    assert torch.isclose(masked_l1(pred, tgt, mask), torch.tensor(1.0))


def test_stft_loss_is_zero_for_identical_signals_and_positive_otherwise():
    x = torch.randn(2, 8192)
    loss = MultiResSTFTLoss()
    assert loss(x, x) < 1e-6
    assert loss(0.5 * x, x) > 0.1


class FrameRenderer(torch.nn.Module):
    """A stand-in vocoder: frame j becomes a 512-sample tone whose amplitude
    is the frame's first mel value, at the frame's F0."""

    def forward(self, mel, f0):
        t = torch.arange(HOP) / 44100.0
        amp = mel[:, 0, :, None]
        tone = torch.sin(2 * torch.pi * f0[..., None] * t)
        return (amp * tone).flatten(1)[:, None, :]


def render(amp: torch.Tensor, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mel = torch.zeros(amp.shape[0], 128, amp.shape[1])
    mel[:, 0] = amp
    return FrameRenderer()(mel, f0)[:, 0], mel


def test_wave_loss_compares_the_rendered_crop_with_the_matching_audio():
    torch.manual_seed(0)
    amp = torch.rand(3, 100)
    f0 = torch.full((3, 100), 440.0)
    audio, mel = render(amp, f0)
    loss = WaveLoss(FrameRenderer(), frames=32, batch=8)
    stft, _ = loss(mel, f0, audio, torch.ones(3, 100))
    assert stft < 1e-4  # aligned: frame j <-> samples [j * hop, (j + 1) * hop)
    shifted = torch.roll(audio, HOP // 2, dims=-1)
    stft_off, _ = loss(
        mel, f0, shifted, torch.ones(3, 100), torch.Generator().manual_seed(0)
    )
    assert stft_off > 10 * stft


def test_wave_loss_skips_clips_too_short_for_a_crop():
    amp, f0 = torch.rand(2, 100), torch.full((2, 100), 440.0)
    audio, mel = render(amp, f0)
    mask = torch.ones(2, 100)
    mask[:, 40:] = 0  # 40 real frames < 32 + 2 * 8 context
    assert WaveLoss(FrameRenderer(), frames=32, batch=8)(mel, f0, audio, mask) is None
    mask[1] = 1
    out = WaveLoss(FrameRenderer(), frames=32, batch=8)(mel, f0, audio, mask)
    assert out is not None


@pytest.mark.models
def test_pretrained_vocoder_loads_and_follows_f0():
    import numpy as np
    import pyworld

    from frontend.mel import LogMel
    from tests.helpers import cents, tone
    from vocoder import load_generator

    gen = load_generator("cpu")
    x = torch.from_numpy(tone(220.0, 1.0, 44100, harmonics=12, amp=0.3))[None]
    x = x[:, : x.shape[-1] // HOP * HOP]
    mel = LogMel()(x)
    for ratio in (1.0, 2.0 ** (5 / 12)):
        with torch.no_grad():
            y = gen(mel, torch.full((1, mel.shape[-1]), 220.0 * ratio))[0, 0]
        assert y.shape == (x.shape[-1],)
        est, _ = pyworld.harvest(y.double().numpy(), 44100, frame_period=10.0)
        est = est[10:-10]
        assert abs(cents(float(np.median(est[est > 0])), 220.0 * ratio)) < 10
