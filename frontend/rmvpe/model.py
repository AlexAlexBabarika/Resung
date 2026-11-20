"""RMVPE pitch estimator (Wei et al., 2023), network and input features.

Vendored from RVC-Project/Retrieval-based-Voice-Conversion-WebUI,
infer/rmvpe.py (MIT, see LICENSE in this folder), cut down to the PyTorch
path. Layer names are unchanged so the published rmvpe.pt loads as is.
"""

import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "shortcut"):
            return self.conv(x) + self.shortcut(x)
        return self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum
                )
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_channel = out_channels

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            skips.append(t)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)]
        )
        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [ConvBlockRes(out_channels * 2, out_channels, momentum)]
        )
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.intermediate(x)
        return self.decoder(x, skips)


class E2E(nn.Module):
    """log-mel [B, 128, T] -> pitch salience [B, T, 360] (20-cent bins)."""

    def __init__(
        self,
        n_blocks=4,
        n_gru=1,
        kernel_size=(2, 2),
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class RMVPEMel(nn.Module):
    """RMVPE's input: 16 kHz, 1024-point STFT, hop 160, 128 HTK mel bands
    from 30 Hz to 8 kHz, natural log. Frame i is centred on sample 160 * i."""

    mel_basis: torch.Tensor
    window: torch.Tensor

    def __init__(self):
        super().__init__()
        basis = librosa_mel(
            sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True
        )
        self.register_buffer(
            "mel_basis", torch.from_numpy(basis).float(), persistent=False
        )
        self.register_buffer("window", torch.hann_window(1024), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=1024,
            hop_length=160,
            win_length=1024,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()
        return torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))
