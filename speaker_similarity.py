"""Speaker similarity for evaluation: how much an output still sounds like
the singer of the input.

    score = SpeakerSimilarity(device)
    score(original16, output16)   # cosine of the two x-vectors, 1 = same

The model is WavLM-base-plus-sv (Microsoft, x-vectors on WavLM, trained for
speaker verification on VoxCeleb1), and on purpose not the ECAPA model whose
embedding conditions the decoder (frontend/timbre.py): scored with its own
conditioning model, the decoder would be marked on reproducing its input.

A cosine has no absolute meaning for singing. Read it next to two anchors
measured on the same material: the same voice on other audio, and other
voices (scripts/voice_check.py prints both).
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import WavLMForXVector
from transformers.utils import logging as hf_logging

from config import FrontendConfig

cfg = FrontendConfig()

REPO = "microsoft/wavlm-base-plus-sv"
REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
# The model takes the raw waveform. Its level is fixed here so the score does
# not depend on how loud a clip is.
RMS = 0.05
# WavLM's attention grows with the square of the length, so long audio is
# embedded in pieces and the pieces' embeddings averaged.
MAX_SECONDS = 20.0
MIN_SECONDS = 2.0


def pieces(n: int, max_len: int, min_len: int) -> list[tuple[int, int]]:
    """(start, end) of consecutive pieces of at most max_len samples covering
    n samples. A last piece shorter than min_len is dropped, unless it is the
    only one."""
    out = [(a, min(a + max_len, n)) for a in range(0, n, max_len)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_len:
        out.pop()
    return out


class SpeakerSimilarity:
    def __init__(self, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            model, info = WavLMForXVector.from_pretrained(
                REPO, revision=REVISION, output_loading_info=True
            )
        finally:
            hf_logging.set_verbosity(verbosity)
        if info["missing_keys"] or info["unexpected_keys"]:
            raise RuntimeError(f"{REPO} did not load cleanly: {info}")
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, audio16: np.ndarray | torch.Tensor) -> torch.Tensor:
        """One clip [N] at 16 kHz -> its unit x-vector [512], on the CPU.
        One clip at a time: this model normalises over time (group norm), so
        zero-padding a batch would move the embeddings."""
        x = torch.as_tensor(audio16, dtype=torch.float32).flatten()
        sr = cfg.analysis_sr
        total = x.new_zeros(self.model.config.xvector_output_dim)
        for a, b in pieces(len(x), round(MAX_SECONDS * sr), round(MIN_SECONDS * sr)):
            piece = x[a:b]
            piece = piece * (RMS / piece.pow(2).mean().sqrt().clamp(min=1e-8))
            emb = self.model(piece[None].to(self.device)).embeddings[0].cpu()
            total += (b - a) * F.normalize(emb, dim=-1)
        return F.normalize(total, dim=-1)

    def __call__(
        self, a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor
    ) -> float:
        return float(self.embed(a) @ self.embed(b))
