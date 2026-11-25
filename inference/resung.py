"""Resung's vocal engine: shift one voice by re-generating it at the new pitch.

    shifter = ResungShifter(ckpt, device)
    shifted = shifter.shift(mono_44k, semitones)

The steps mirror training: the voice is loudness-normalised to -23 LUFS
(undone at the end), RMVPE tracks F0 at 16 kHz, ContentVec reads the
unshifted voice, and the decoder predicts the vocoder's mel from that content
with F0 x 2^(s/12), the clip's loudness and its timbre. The decoder runs over
overlapping windows of the training length (attention is only trained on
2.6 s), whose mels are cross-faded; PC-NSF-HiFiGAN then renders the whole
mel at the shifted F0 in one pass.

Formants are whatever the decoder learned for this voice at the new pitch;
there is no separate formant control yet.
"""

from pathlib import Path

import numpy as np
import torch

from config import FrontendConfig
from frontend.audio import analysis_len, normalization_gain, to_analysis_sr
from frontend.content import ContentVec
from frontend.f0 import RMVPE, f0_at, interp_unvoiced, mel_frame_times
from frontend.mel import LogMel, Loudness
from frontend.timbre import Timbre
from model import load_decoder
from vocoder import load_generator

fe = FrontendConfig()
F0_CHUNK = 60 * fe.analysis_sr  # RMVPE in 60 s pieces (a multiple of its hop)
WINDOW = fe.clip_frames  # decoder window = the training crop, 224 frames
OVERLAP = 64  # frames cross-faded between neighbouring windows
TIMBRE_SEG = round(fe.segment_seconds * fe.analysis_sr)  # 5.12 s, as in the cache


def find_resung_ckpt(root: Path = Path("stage1")) -> Path | None:
    for name in ("final.pt", "best.pt"):
        if (root / name).exists():
            return root / name
    steps = sorted(
        (root / "checkpoints").glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    ) + sorted(root.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return steps[-1] if steps else None


class ResungShifter:
    def __init__(self, ckpt: Path, device: torch.device, verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self.decoder = load_decoder(ckpt, device)
        self.content = ContentVec().to(device)
        self.rmvpe = RMVPE(device)
        self.timbre = Timbre(device)
        self.loudness = Loudness().to(device)
        self.logmel = LogMel().to(device)
        self.vocoder = load_generator(device)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @torch.inference_mode()
    def track_f0(self, x16: np.ndarray) -> np.ndarray:
        """RMVPE over a 16 kHz signal of any length: Hz per 10 ms, 0 = unvoiced."""
        parts = []
        for start in range(0, len(x16), F0_CHUNK):
            chunk = torch.from_numpy(x16[start : start + F0_CHUNK])[None]
            f0, _ = self.rmvpe(chunk)
            f0 = f0[0].cpu().numpy()
            parts.append(
                f0 if start + F0_CHUNK >= len(x16) else f0[: F0_CHUNK // fe.f0_hop]
            )
        return np.concatenate(parts)

    @torch.inference_mode()
    def _timbre(self, x16: np.ndarray, f0_track: np.ndarray) -> torch.Tensor:
        # Mean ECAPA embedding over the mostly-sung 5.12 s stretches, each
        # embedded unpadded, as the training cache did.
        per_seg = TIMBRE_SEG // fe.f0_hop
        embs = []
        for i, start in enumerate(range(0, len(x16) - TIMBRE_SEG + 1, TIMBRE_SEG)):
            if (f0_track[i * per_seg : (i + 1) * per_seg] > 0).mean() >= 0.5:
                seg = torch.from_numpy(x16[start : start + TIMBRE_SEG])[None]
                embs.append(self.timbre(seg)[0])
        if not embs:
            embs = [self.timbre(torch.from_numpy(x16)[None])[0]]
        self._log(f"  timbre from {len(embs)} sung stretches of 5.12 s")
        return torch.stack(embs).mean(0)

    @torch.inference_mode()
    def _decode(self, x16, f0, uv, loud, timbre, n_frames, batch=16) -> torch.Tensor:
        """Decoder over overlapping training-length windows, cross-faded."""
        hop = fe.hop_length
        win = min(WINDOW, n_frames)
        step = win - OVERLAP if win > OVERLAP else win
        starts = list(range(0, max(n_frames - win, 0) + 1, step))
        if starts[-1] + win < n_frames:
            starts.append(n_frames - win)
        n16 = analysis_len(win * hop)
        ramp = torch.ones(win)
        if win > OVERLAP:
            ramp[:OVERLAP] = torch.linspace(0.0, 1.0, OVERLAP + 2)[1:-1]
            ramp[-OVERLAP:] = torch.linspace(1.0, 0.0, OVERLAP + 2)[1:-1]
        acc = torch.zeros(fe.n_mels, n_frames)
        wsum = torch.zeros(n_frames)
        dev = self.device
        for b0 in range(0, len(starts), batch):
            ks = starts[b0 : b0 + batch]
            crops = []
            for k in ks:
                s16 = analysis_len(k * hop)
                c = x16[s16 : s16 + n16]
                crops.append(np.pad(c, (0, n16 - len(c))))
            content = self.content(torch.from_numpy(np.stack(crops)).to(dev))
            sl = lambda t: torch.stack([t[k : k + win] for k in ks]).to(dev)  # noqa: E731
            mels = self.decoder(
                content,
                sl(f0),
                sl(uv),
                sl(loud),
                timbre[None].expand(len(ks), -1).to(dev),
                torch.ones(len(ks), win, device=dev),
            ).cpu()
            for k, m in zip(ks, mels):
                w = ramp.clone()
                if k == 0:
                    w[:OVERLAP] = 1.0
                if k + win >= n_frames:
                    w[-OVERLAP:] = 1.0
                acc[:, k : k + win] += m * w
                wsum[k : k + win] += w
            self._log(f"  decoded windows {b0 + len(ks)}/{len(starts)}")
        return acc / wsum.clamp(min=1e-6)

    def _prepare(self, mono: np.ndarray):
        """The voice as training saw it: loudness-normalised and padded to
        whole frames. Returns it with the gain applied, the true length, the
        16 kHz copy and RMVPE's track."""
        gain = normalization_gain(mono)
        x = (mono * gain).astype(np.float32)
        n = len(x)
        x = np.pad(x, (0, -n % fe.hop_length))
        x16 = to_analysis_sr(x)
        self._log("  F0 (RMVPE)")
        return x, gain, n, x16, self.track_f0(x16)

    def _vocode(self, mel: torch.Tensor, f0: torch.Tensor) -> np.ndarray:
        self._log("  vocoder")
        try:
            y = self.vocoder(mel[None].to(self.device), f0[None].to(self.device))[0, 0]
        except RuntimeError as e:  # e.g. an MPS kernel limit on very long inputs
            self._log(f"  vocoder on {self.device} failed ({e}); retrying on CPU")
            y = self.vocoder.cpu()(mel[None].cpu(), f0[None])[0, 0]
            self.vocoder.to(self.device)
        return y.cpu().numpy()

    @torch.inference_mode()
    def shift(
        self, mono: np.ndarray, semitones: float, timbre: torch.Tensor | None = None
    ) -> np.ndarray:
        """mono: [samples] at 44.1 kHz -> the same voice, shifted, same length.
        timbre: an embedding from timbre_of(), to sing with another voice's
        instead of this one's own."""
        x, gain, n, x16, f0_track = self._prepare(mono)
        n_frames = len(x) // fe.hop_length
        f0c, uv = interp_unvoiced(f0_at(f0_track, mel_frame_times(n_frames)))
        f0 = torch.from_numpy(f0c * 2.0 ** (semitones / 12.0))
        loud = self.loudness(torch.from_numpy(x)[None].to(self.device))[0].cpu()
        if timbre is None:
            timbre = self._timbre(x16, f0_track)

        self._log(f"  decoder over {n_frames} frames")
        mel = self._decode(x16, f0, torch.from_numpy(uv), loud, timbre, n_frames)
        return (self._vocode(mel, f0)[:n] / gain).astype(np.float32)

    @torch.inference_mode()
    def timbre_of(self, mono: np.ndarray) -> torch.Tensor:
        """The timbre embedding shift() would use for this voice."""
        _, _, _, x16, f0_track = self._prepare(mono)
        return self._timbre(x16, f0_track)

    @torch.inference_mode()
    def copy_synthesis(self, mono: np.ndarray) -> np.ndarray:
        """The voice's own mel and F0 through the vocoder, with no decoder:
        the best shift(mono, 0) could do."""
        x, gain, n, _, f0_track = self._prepare(mono)
        mel = self.logmel(torch.from_numpy(x)[None].to(self.device))[0]
        f0c, _ = interp_unvoiced(f0_at(f0_track, mel_frame_times(mel.shape[-1])))
        return (self._vocode(mel, torch.from_numpy(f0c))[:n] / gain).astype(np.float32)
