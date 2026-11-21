"""PC-NSF-HiFiGAN, the 44.1 kHz singing vocoder Resung decodes into."""

from vocoder.load import checkpoint_path, load_generator, load_state
from vocoder.nsf_hifigan import Generator

__all__ = ["Generator", "checkpoint_path", "load_generator", "load_state"]
