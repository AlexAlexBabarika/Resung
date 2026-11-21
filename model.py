"""Resung's decoder: what is sung, at which pitch, how loud, by whom -> mel.

Inputs, all on the vocoder's frames except the content:
    content   [B, Tc, 768]  ContentVec at 50 Hz (frame t centred at
                            (320 t + 200) / 16000 s)
    f0        [B, T]        Hz, unvoiced frames interpolated
    uv        [B, T]        1 = unvoiced
    loudness  [B, T]        dB (A-weighted)
    timbre    [B, 192]      ECAPA embedding, one per clip
    mask      [B, T]        1 = real frame
Output: natural-log mel [B, 128, T] for PC-NSF-HiFiGAN.

The content goes through a narrow projection (a bottleneck that squeezes out
leftover pitch), is interpolated onto the mel frames, and joins Fourier
features of log-F0 and the loudness. A stack of conv + self-attention blocks
follows; the timbre enters every block through adaptive layer norms whose
residual gates start at zero, so the stack starts as the identity.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FrontendConfig, ModelConfig
from frontend.content import content_frame_times

F0_REF_HZ = 440.0
F0_FLOOR_HZ = 30.0


def mel_frame_centres(
    n_frames: int, fe: FrontendConfig = FrontendConfig()
) -> torch.Tensor:
    """Acoustic centre of each vocoder frame, in seconds: frame j covers
    samples [j * hop - 768, j * hop + 1280) at 44.1 kHz."""
    return (torch.arange(n_frames) + 0.5) * fe.hop_length / fe.sr


def resample_frames(
    x: torch.Tensor, src_t: torch.Tensor, dst_t: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of x [B, Ts, C] from times src_t [Ts] to dst_t
    [Td]; times outside the source hold the end frames.

    Indices and weights are computed in float32 whatever x's dtype: in
    bfloat16 (autocast) the time grids themselves round off, and the content
    was read up to 14 ms off in a 2.6 s crop, 120 ms off in a 30 s window."""
    src_t = src_t.to(x.device, torch.float32)
    dst_t = dst_t.to(x.device, torch.float32)
    hi = torch.searchsorted(src_t, dst_t).clamp(1, len(src_t) - 1)
    lo = hi - 1
    w = ((dst_t - src_t[lo]) / (src_t[hi] - src_t[lo])).clamp(0.0, 1.0)
    w = w.to(x.dtype)[None, :, None]
    return x[:, lo] * (1 - w) + x[:, hi] * w


class AdaNorm(nn.Module):
    """LayerNorm whose scale and shift come from the conditioning vector,
    plus a gate for the residual branch. All three start at zero."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift, gate = self.proj(cond).unsqueeze(1).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


def rotary(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on [B, H, T, Dh]: relative positions, so
    attention behaves the same on windows longer than the training crop.
    The rotation runs in float32 and the result returns in x's dtype."""
    t, dh = x.shape[-2], x.shape[-1]
    freqs = 1.0 / (
        10000.0 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    )
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (
            self.qkv(x)
            .view(b, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = rotary(q), rotary(k)
        attn_mask = key_mask[:, None, None, :].bool()  # True = attend
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class ConvModule(nn.Module):
    """Pointwise GLU, depthwise conv over time, pointwise out."""

    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.pw_in = nn.Linear(dim, 2 * dim)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pw_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = F.glu(self.pw_in(x), dim=-1) * mask[..., None]  # padding stays silent
        x = F.silu(self.dw(x.transpose(1, 2))).transpose(1, 2)
        return self.drop(self.pw_out(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_attn = AdaNorm(cfg.dim, cfg.cond_dim)
        self.attn = Attention(cfg.dim, cfg.heads, cfg.dropout)
        self.n_conv = AdaNorm(cfg.dim, cfg.cond_dim)
        self.conv = ConvModule(cfg.dim, cfg.conv_kernel, cfg.dropout)
        self.n_ff = AdaNorm(cfg.dim, cfg.cond_dim)
        self.ff = FeedForward(cfg.dim, cfg.ff_mult, cfg.dropout)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        h, g = self.n_attn(x, cond)
        x = x + g * self.attn(h, mask)
        h, g = self.n_conv(x, cond)
        x = x + g * self.conv(h, mask)
        h, g = self.n_ff(x, cond)
        return x + g * self.ff(h)


class ResungDecoder(nn.Module):
    def __init__(
        self, cfg: ModelConfig = ModelConfig(), fe: FrontendConfig = FrontendConfig()
    ):
        super().__init__()
        self.cfg, self.fe = cfg, fe
        self.content_in = nn.Sequential(
            nn.Linear(cfg.content_dim, cfg.content_bottleneck),
            nn.Linear(cfg.content_bottleneck, cfg.dim),
        )
        # log2(F0 / 440) spans about -3.9 (30 Hz) to +2.5 (2.5 kHz). Periods
        # run from 16 octaves down to 2^-6 octave (19 cents); finer ones would
        # only encode RMVPE's few cents of jitter.
        self.register_buffer(
            "f0_freqs",
            math.pi * 2.0 ** torch.arange(cfg.f0_fourier, dtype=torch.float32) / 8.0,
            persistent=False,
        )
        self.f0_in = nn.Linear(2 * cfg.f0_fourier + 2, cfg.dim)  # + log-F0, uv
        self.loud_in = nn.Linear(1, cfg.dim)
        self.timbre = nn.Sequential(
            nn.Linear(fe.timbre_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.n_out = AdaNorm(cfg.dim, cfg.cond_dim)
        self.out = nn.Linear(cfg.dim, fe.n_mels)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.constant_(self.out.bias, -6.0)  # about a quiet frame's log-mel

    f0_freqs: torch.Tensor

    def forward(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        uv: torch.Tensor,
        loudness: torch.Tensor,
        timbre: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        n = f0.shape[-1]
        c = self.content_in(content)
        c = resample_frames(
            c, content_frame_times(content.shape[1]), mel_frame_centres(n, self.fe)
        )
        lf0 = torch.log2(f0.clamp(min=F0_FLOOR_HZ) / F0_REF_HZ)[..., None]
        ang = lf0 * self.f0_freqs
        f0_feat = torch.cat([ang.sin(), ang.cos(), lf0, uv[..., None]], dim=-1)
        loud = ((loudness + 40.0) / 20.0)[..., None]
        x = (c + self.f0_in(f0_feat) + self.loud_in(loud)) * mask[..., None]
        cond = self.timbre(F.normalize(timbre, dim=-1))
        for block in self.blocks:
            x = block(x, cond, mask)
        h, _ = self.n_out(x, cond)
        # The projection to the mel runs in float32 even under autocast:
        # in bf16, log-mels around -6 round to steps of 1/32.
        with torch.autocast(device_type=h.device.type, enabled=False):
            mel = self.out(h.float())
        return mel.transpose(1, 2)  # [B, n_mels, T]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_decoder(path, device="cpu", use_ema: bool = True) -> ResungDecoder:
    """A trained decoder from a train.py checkpoint, EMA weights by default."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    decoder = ResungDecoder(ModelConfig(**ck["model_config"]))
    decoder.load_state_dict(ck["decoder"])
    if use_ema:
        with torch.no_grad():
            for n, p in decoder.named_parameters():
                p.copy_(ck["ema"]["shadow"][n])
    return decoder.eval().to(device)
