"""Fetches and loads the PC-NSF-HiFiGAN 2025.02 checkpoint.

The weights come from OpenVPI's official SingingVocoders release, under
CC BY-NC-SA 4.0. They are downloaded into the local cache, never committed.
Redistributing them, fine-tuned or not, requires a copy of the release's
NOTICE.txt (kept next to the checkpoint), a link to the licence, and a note
saying what was changed.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch

from vocoder.nsf_hifigan import Generator

RELEASE_URL = (
    "https://github.com/openvpi/SingingVocoders/releases/download/v1.0.0/"
    "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip"
)
RELEASE_SHA256 = "636845f96066ab95c188c677713f4359ae53eae9669a7500d926c97174c3590c"
CKPT_NAME = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt"


def cache_dir() -> Path:
    root = Path(os.environ.get("RESUNG_CACHE", Path.home() / ".cache" / "resung"))
    return root / "pc_nsf_hifigan_2025.02"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_path() -> Path:
    """Download (once), verify and unpack the release. Under torchrun, call it
    on rank 0 before the other ranks."""
    out = cache_dir()
    ckpt = out / CKPT_NAME
    if ckpt.exists():
        return ckpt
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        zpath = Path(tmp) / "release.zip"
        print(f"downloading the vocoder from {RELEASE_URL}")
        with urllib.request.urlopen(RELEASE_URL) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        digest = _sha256(zpath)
        if digest != RELEASE_SHA256:
            raise RuntimeError(
                f"vocoder download has sha256 {digest}, expected {RELEASE_SHA256}"
            )
        with zipfile.ZipFile(zpath) as z:
            for name in (CKPT_NAME, "NOTICE.txt", "STATEMENTS.txt"):
                z.extract(name, tmp)
                os.replace(Path(tmp) / name, out / name)
    return ckpt


def load_state() -> dict[str, torch.Tensor]:
    """The release's full state dict: generator.* and discriminator.*"""
    ckpt = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    return ckpt["state_dict"]


def _to_parametrized(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Old-style weight norm (weight_g, weight_v) -> parametrizations API.
    out = {}
    for k, v in state.items():
        if k.endswith(".weight_g"):
            k = k[: -len("weight_g")] + "parametrizations.weight.original0"
        elif k.endswith(".weight_v"):
            k = k[: -len("weight_v")] + "parametrizations.weight.original1"
        out[k] = v
    return out


def load_generator(
    device: str | torch.device = "cpu", trainable: bool = False
) -> Generator:
    """The pretrained generator. Frozen by default: weight norm folded in,
    eval mode, no gradients to its weights (they still flow to its input)."""
    state = load_state()
    gen_state = {
        k[len("generator.") :]: v
        for k, v in state.items()
        if k.startswith("generator.")
    }
    gen = Generator()
    gen.load_state_dict(_to_parametrized(gen_state), strict=True)
    if not trainable:
        gen.remove_weight_norm()
        gen.eval()
        gen.requires_grad_(False)
    return gen.to(device)
