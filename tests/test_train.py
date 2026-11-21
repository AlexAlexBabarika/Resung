"""One stage-1 step on the synthetic cache from test_data (needs the models)."""

import pytest
import torch

from config import TrainConfig
from data import ResungDataset
from tests.test_data import cache  # noqa: F401  (fixture)

pytestmark = pytest.mark.models


def test_a_training_step_updates_only_the_decoder(cache):  # noqa: F811
    from train import Stage1

    cfg = TrainConfig()
    cfg.wave_batch, cfg.wave_frames = 2, 32
    stage = Stage1(cfg, torch.device("cpu"))
    ds = ResungDataset("train", dataset_keys=["vctk"], cache_dir=cache)
    batch = {k: torch.stack([ds[i][k] for i in range(2)]) for k in ds[0]}
    loss, parts = stage.losses(batch)
    assert torch.isfinite(loss) and {"mel", "stft", "wave_mel", "loss"} <= parts.keys()
    loss.backward()
    assert all(
        p.grad is not None for p in stage.decoder.parameters() if p.requires_grad
    )
    frozen = [*stage.content.parameters(), *stage.wave.generator.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in frozen)
