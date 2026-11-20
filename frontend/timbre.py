"""Timbre: an ECAPA-TDNN speaker embedding (SpeechBrain, trained on VoxCeleb).

Used for conditioning only. Evaluation measures speaker similarity with a
different model (WavLM-base-plus-sv), so the score isn't biased towards the
embedding the decoder was trained on.
"""

import torch
from huggingface_hub import snapshot_download

from config import FrontendConfig

cfg = FrontendConfig()

TIMBRE_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


class Timbre:
    """16 kHz audio [B, N] -> embeddings [B, 192].

    Every row must be real audio. Zero-padding moves the embedding even when
    SpeechBrain is told the true lengths (cosine 0.97 with a third of a 3 s
    clip padded), so embed unequal clips one at a time.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier

        local = snapshot_download(cfg.timbre_repo, revision=TIMBRE_REVISION)
        self.device = torch.device(device)
        model = EncoderClassifier.from_hparams(
            source=local, savedir=local, run_opts={"device": str(self.device)}
        )
        assert model is not None
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(self, audio16: torch.Tensor) -> torch.Tensor:
        audio16 = audio16.to(self.device, torch.float32)
        return self.model.encode_batch(audio16).squeeze(1)
