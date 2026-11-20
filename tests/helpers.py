import numpy as np


def tone(
    freq: float, seconds: float, sr: int, harmonics: int = 1, amp: float = 0.5
) -> np.ndarray:
    """A harmonic tone with 1/h partials, peak-normalised to amp."""
    t = np.arange(int(round(seconds * sr))) / sr
    x = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, harmonics + 1))
    return (amp * x / np.max(np.abs(x))).astype(np.float32)


def cents(a: float, b: float) -> float:
    return float(1200.0 * np.log2(a / b))
