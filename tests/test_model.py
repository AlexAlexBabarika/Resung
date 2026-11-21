import torch

from config import ModelConfig
from frontend.content import content_frame_times
from model import ResungDecoder, mel_frame_centres, resample_frames, rotary

SMALL = ModelConfig(dim=64, layers=2, heads=4, cond_dim=32, content_bottleneck=32)


def inputs(b=2, t=60, tc=36):
    torch.manual_seed(0)
    return {
        "content": torch.randn(b, tc, 768),
        "f0": 150.0 + 100.0 * torch.rand(b, t),
        "uv": (torch.rand(b, t) > 0.8).float(),
        "loudness": -30.0 + 10.0 * torch.rand(b, t),
        "timbre": torch.randn(b, 192),
        "mask": torch.ones(b, t),
    }


def open_gates(model: torch.nn.Module) -> None:
    # The residual gates start at zero; open them so every block matters.
    torch.manual_seed(1)
    for m in model.modules():
        if m.__class__.__name__ == "AdaNorm":
            torch.nn.init.normal_(m.proj.weight, std=0.1)
            torch.nn.init.normal_(m.proj.bias, std=0.5)


def test_resample_frames_interpolates_linearly_and_holds_the_ends():
    x = torch.tensor([[[0.0], [10.0], [20.0]]])
    src = torch.tensor([0.0, 1.0, 2.0])
    out = resample_frames(x, src, torch.tensor([-1.0, 0.25, 1.5, 3.0]))
    assert torch.allclose(out[0, :, 0], torch.tensor([0.0, 2.5, 15.0, 20.0]))


def test_content_and_mel_frames_cover_the_same_span():
    # A 2.6 s training crop: 129 content frames, 224 mel frames.
    c, m = content_frame_times(129), mel_frame_centres(224)
    assert abs(float(c[0]) - float(m[0])) < 0.01
    # ContentVec's 400-sample receptive field stops ~1 hop short of the
    # last mel frame; resample_frames holds the end frame there.
    assert abs(float(c[-1]) - float(m[-1])) < 0.03  # 1.5 content frames


def test_rotary_keeps_norms():
    x = torch.randn(2, 3, 10, 16)
    assert torch.allclose(rotary(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_decoder_shape_and_initial_output():
    model = ResungDecoder(SMALL).eval()
    out = model(**inputs())
    assert out.shape == (2, 128, 60)
    assert abs(float(out.mean()) + 6.0) < 0.5  # starts near a quiet frame


def test_padded_frames_do_not_leak_into_real_ones():
    model = ResungDecoder(SMALL).eval()
    open_gates(model)
    a = inputs()
    a["mask"][:, 40:] = 0.0
    b = {k: v.clone() for k, v in a.items()}
    b["f0"][:, 40:] = 900.0
    b["loudness"][:, 40:] = 0.0
    b["uv"][:, 40:] = 1.0
    with torch.no_grad():
        ya, yb = model(**a), model(**b)
    assert torch.allclose(ya[..., :40], yb[..., :40], atol=1e-5)


def test_every_input_reaches_the_output():
    model = ResungDecoder(SMALL).eval()
    open_gates(model)
    base = inputs()
    with torch.no_grad():
        ref = model(**base)
        for key, change in [
            ("content", lambda v: v + 1.0),
            ("f0", lambda v: v * 2.0),
            ("loudness", lambda v: v + 10.0),
            ("timbre", lambda v: torch.randn_like(v)),
        ]:
            alt = dict(base, **{key: change(base[key])})
            # Any real dependence is orders of magnitude above float noise.
            assert (model(**alt) - ref).abs().mean() > 1e-4, key


def test_resample_frames_reads_the_right_time_in_bf16():
    # The GPU run autocasts to bfloat16. A one-hot basis is exact in bf16,
    # so the output rows are the interpolation weights themselves.
    for seconds in (2.6, 30.0):
        tc = int((seconds * 16000 - 400) // 320 + 1)
        src, dst = (
            content_frame_times(tc),
            mel_frame_centres(int(seconds * 44100 // 512)),
        )
        eye = torch.eye(tc)[None]
        w32 = resample_frames(eye, src, dst)[0]
        w16 = resample_frames(eye.bfloat16(), src, dst)[0].float()
        # The time each output frame reads (weights normalised: bf16 rounds
        # their sum by up to 0.2%, a level change, not a timing one).
        t32 = (w32 @ src) / w32.sum(1)
        t16 = (w16 @ src) / w16.sum(1)
        assert (t16 - t32).abs().max() < 1e-4  # seconds; was 14 ms / 120 ms


def test_decoder_under_bf16_autocast_stays_close_to_fp32():
    model = ResungDecoder(SMALL).eval()
    open_gates(model)
    a = inputs()
    with torch.no_grad():
        y32 = model(**a)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y16 = model(**a)
    # The predicted mel stays float32: bf16 would round log-mels near -6 to
    # steps of 1/32.
    assert y16.dtype == torch.float32 and torch.isfinite(y16).all()
    assert (y32 - y16).abs().mean() < 0.01  # log-mel units; a good fit is ~0.25
