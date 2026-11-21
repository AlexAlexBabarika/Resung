"""Stage-1 losses: the decoder's mel, and what the frozen vocoder makes of it."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontend.mel import LogMel


def masked_l1(
    pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """pred, tgt: [B, C, T]; mask: [B, T], 1 on real frames."""
    m = mask[:, None, :].to(pred.dtype)
    return ((pred - tgt).abs() * m).sum() / (m.sum() * pred.shape[1]).clamp(min=1.0)


class MultiResSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1, averaged over resolutions.
    Magnitudes only: the vocoder chooses its own phases."""

    def __init__(self, resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.resolutions = resolutions
        for n_fft, _ in resolutions:
            self.register_buffer(
                f"win_{n_fft}", torch.hann_window(n_fft), persistent=False
            )

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        loss = pred.new_zeros(())
        for n_fft, hop in self.resolutions:
            win = getattr(self, f"win_{n_fft}")
            p = (
                torch.stft(pred, n_fft, hop, window=win, return_complex=True)
                .abs()
                .clamp(min=1e-5)
            )
            t = (
                torch.stft(tgt, n_fft, hop, window=win, return_complex=True)
                .abs()
                .clamp(min=1e-5)
            )
            sc = torch.linalg.norm(t - p) / torch.linalg.norm(t).clamp(min=1e-5)
            loss = loss + sc + F.l1_loss(p.log(), t.log())
        return loss / len(self.resolutions)


class WaveLoss(nn.Module):
    """Renders a crop of the predicted mel through the frozen vocoder and
    compares it with the audio: multi-resolution STFT, and the vocoder-mel
    L1 of the output (the term HiFi-GAN training uses). Mel frame j renders
    samples [j * hop, (j + 1) * hop), checked by scripts/vocoder_check.py."""

    CONTEXT = 8  # extra frames each side; their audio is discarded

    def __init__(self, generator: nn.Module, frames: int, batch: int, hop: int = 512):
        super().__init__()
        self.generator = generator
        self.frames, self.batch, self.hop = frames, batch, hop
        self.stft = MultiResSTFTLoss()
        self.logmel = LogMel()

    def forward(
        self,
        pred_mel: torch.Tensor,
        f0: torch.Tensor,
        audio: torch.Tensor,
        mask: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """pred_mel [B, n_mels, T], f0 [B, T] Hz, audio [B, T * hop],
        mask [B, T]. Returns (stft, mel) losses, or None if no clip in the
        batch is long enough for a crop."""
        ctx, hop, frames = self.CONTEXT, self.hop, self.frames
        valid = mask.sum(-1).long()
        rows = torch.nonzero(valid >= frames + 2 * ctx).flatten()[: self.batch]
        if len(rows) == 0:
            return None
        span = frames + 2 * ctx
        mels, f0s, tgts = [], [], []
        for r in rows.tolist():
            hi = int(valid[r]) - frames - ctx
            a = ctx + int(torch.randint(0, hi - ctx + 1, (1,), generator=rng))
            mels.append(pred_mel[r, :, a - ctx : a - ctx + span])
            f0s.append(f0[r, a - ctx : a - ctx + span])
            tgts.append(audio[r, a * hop : (a + frames) * hop])
        mel, f0c, tgt = (
            torch.stack(mels).float(),
            torch.stack(f0s).float(),
            torch.stack(tgts).float(),
        )
        with torch.autocast(device_type=mel.device.type, enabled=False):
            out = self.generator(mel, f0c)[:, 0, ctx * hop : (ctx + frames) * hop]
            stft = self.stft(out, tgt)
            wave_mel = F.l1_loss(self.logmel(out), self.logmel(tgt))
        return stft, wave_mel
