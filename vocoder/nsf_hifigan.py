"""PC-NSF-HiFiGAN generator (mini-NSF source).

Vendored from openvpi/SingingVocoders, models/nsf_HiFigan/models.py (MIT, see
LICENSE in this folder), reduced to the mini-NSF path the 2025.02 release
uses. Layer names are unchanged so the released checkpoint loads strictly.
Weight norm uses the parametrizations API; load_generator() maps the
checkpoint's weight_g / weight_v onto it.

The weights are not code: OpenVPI publishes them under CC BY-NC-SA 4.0
(see vocoder/load.py).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)
        m.bias.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in [*self.convs1, *self.convs2]:
            remove_parametrizations(conv, "weight")


class Generator(nn.Module):
    """log-mel [B, num_mels, T] (natural log) and F0 [B, T] in Hz on the
    same frames -> audio [B, 1, T * prod(upsample_rates)] in [-1, 1]."""

    def __init__(
        self,
        num_mels=128,
        sampling_rate=44100,
        upsample_rates=(8, 8, 2, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4, 4),
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # mini-NSF: the sine source runs at the rate after the first two
        # upsampling stages and joins the network there.
        self.source_sr = sampling_rate / int(np.prod(upsample_rates[2:]))
        self.upp = int(np.prod(upsample_rates[:2]))
        self.hop = int(np.prod(upsample_rates))

        self.conv_pre = weight_norm(
            Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)
        )
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch //= 2
            self.ups.append(
                weight_norm(ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            )
            for k_res, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k_res, d))
            if i == 1:
                self.source_conv = Conv1d(1, ch, 1)
                self.source_conv.apply(init_weights)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def fastsinegen(self, f0: torch.Tensor) -> torch.Tensor:
        # Phase-continuous sine at F0, with F0 ramped linearly inside each
        # frame. Kept in float32: the phase accumulates over the whole clip.
        f0 = f0.float()
        n = torch.arange(1, self.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / self.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / self.upp
        rad2 = torch.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1)
        return torch.sin(2 * np.pi * rad)

    def forward(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        har_source = self.fastsinegen(f0)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            if i == 1:
                x = x + self.source_conv(har_source)
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for m in [*self.ups, self.conv_pre, self.conv_post]:
            remove_parametrizations(m, "weight")
        for block in self.resblocks:
            block.remove_weight_norm()
