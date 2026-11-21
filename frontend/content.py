"""ContentVec features: what is sung, with little of who sings it.

lengyue233/content-vec-best is a transformers conversion of the fairseq
ContentVec_legacy checkpoint. On 2026-11-20 its 213 weight tensors were
compared with RVC's copy of the original (lj1995/VoiceConversionWebUI,
hubert_base.pt) and matched exactly.

We use the last transformer layer (768-d), as RVC v2 does, and leave out the
final_proj layer that the checkpoint carries for backward compatibility.
"""

import torch
import torch.nn as nn
from transformers import HubertModel
from transformers.utils import logging as hf_logging

from config import FrontendConfig

cfg = FrontendConfig()

HOP = 320  # 20 ms at 16 kHz
RECEPTIVE_FIELD = 400


def content_frames(n_analysis_samples: int) -> int:
    return (n_analysis_samples - RECEPTIVE_FIELD) // HOP + 1


def content_frame_times(n_frames: int) -> torch.Tensor:
    """Centre of each content frame, in seconds from the start of the clip."""
    return (torch.arange(n_frames) * HOP + RECEPTIVE_FIELD / 2) / cfg.analysis_sr


class ContentVec(nn.Module):
    """16 kHz audio [B, N] in [-1, 1] -> features [B, content_frames(N), 768].
    Frozen. Pad a batch with zeros rather than passing an attention mask:
    this model normalises its first conv layer over time (group norm)."""

    def __init__(self):
        super().__init__()
        # transformers would print a load report listing the expected
        # final_proj.* leftovers; the check below inspects the same report.
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            model, info = HubertModel.from_pretrained(
                cfg.content_repo, revision=cfg.content_revision, output_loading_info=True
            )
        finally:
            hf_logging.set_verbosity(verbosity)
        missing = info["missing_keys"]
        unexpected = [
            k for k in info["unexpected_keys"] if not k.startswith("final_proj.")
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"ContentVec checkpoint did not load cleanly: missing={missing}, "
                f"unexpected={unexpected}"
            )
        self.model = model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        # Always frozen, always in eval mode (no SpecAugment masking).
        return super().train(False)

    @torch.no_grad()
    def forward(self, audio16: torch.Tensor) -> torch.Tensor:
        return self.model(audio16).last_hidden_state
