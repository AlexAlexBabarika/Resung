"""Where does Resung lose the singer? Stage 1 moved the pitch of a real song
accurately, but the output no longer sounded like the singer.

    uv run scripts/voice_check.py --ckpt stage1/step_46000.pt \\
        --voice seen EN-Alto-1 datasets/gtsinger/English/EN-Alto-1 \\
        --voice unseen JA-Soprano-1 datasets/gtsinger/Japanese/JA-Soprano-1 \\
        --voice separated song path/to/song_stems/vocals.wav

--voice KIND NAME PATH... names one voice: audio files, or folders of them.
KIND is any label; the summary averages over it (voices the decoder trained
on, clean voices it never heard, vocals separated from a song).

Each voice gives two excerpts that don't overlap, A and B, each the stretch
with the most singing. A goes through the systems below, and each output is
scored against A by speaker similarity (speaker_similarity.py, a cosine):

    same     B: the same voice on other audio. What "the same singer" scores.
    others   the other voices' A, averaged. What "another singer" scores.
    vocoder  A's own mel and F0 through the vocoder: the best the decoder
             could do.
    resung   the decoder at the original pitch, timbre from A, as inference
             does. Well under `vocoder` means the voice is lost with no
             shift at all, so the shift is not what loses it.
    swap     the decoder at the original pitch with the timbre embedding of
             the least similar other voice (the donor). As high as `resung`
             means the decoder ignores the embedding.
    donor    how much the two outputs above sound like the donor's A: with
             A's own timbre, then with the donor's. A rise means the
             embedding carries a voice.
    resung+N, rubberband+N  both shifted by N semitones. The scorer hears
             pitch too, so Rubber Band is the reference for a shifted voice.

Also the log-mel L1 from the vocoder's output to `resung` (recon) and from
`resung` to `swap` (how far the embedding moves the output).

--backing FILE adds a controlled version of the separated case, for clean
voices: A is mixed into the loudest stretch of that instrumental, Demucs
takes it out again, and the decoder gets the result. Still scored against
the clean A:

    separated           Demucs's vocal itself: what separation alone costs
    resung_sep          content and timbre both from the separated vocal,
                        as inference on a song
    resung_sep_content  content from the separated vocal, timbre from clean A
    resung_sep_timbre   content from clean A, timbre from the separated vocal

Writes <out>/<name>_<system>.wav for listening and <out>/results.json.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FrontendConfig  # noqa: E402
from frontend.audio import resample, to_analysis_sr  # noqa: E402
from frontend.f0 import F0_FRAME_S  # noqa: E402

fe = FrontendConfig()
AUDIO_SUFFIXES = {".wav", ".flac"}


def pick_windows(voiced: np.ndarray, win: int) -> tuple[int, int]:
    """Starts of the two non-overlapping windows of `win` frames with the
    most voiced frames; the first returned is the more voiced one."""
    n = len(voiced) - win + 1
    if n < win + 1:
        raise ValueError(f"need two windows of {win} frames, got {len(voiced)} frames")
    c = np.concatenate([[0], np.cumsum(voiced.astype(np.int64))])
    count = c[win:] - c[:-win]  # voiced frames in the window starting at i
    a = int(count.argmax())
    free = count.copy()
    free[max(0, a - win + 1) : a + win] = -1
    return a, int(free.argmax())


def load_voice(paths: list[Path], max_seconds: float) -> np.ndarray:
    """Mono audio at fe.sr from the files (folders are searched), in name
    order, joined up to max_seconds."""
    files = []
    for p in paths:
        found = sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_SUFFIXES)
        files += found if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"no audio under {[str(p) for p in paths]}")
    parts, total = [], 0
    for f in files:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        parts.append(resample(x.mean(axis=1), sr, fe.sr))
        total += len(parts[-1])
        if total >= max_seconds * fe.sr:
            break
    return np.concatenate(parts)[: round(max_seconds * fe.sr)]


def loudest(x: np.ndarray, n: int, step: int = fe.sr) -> np.ndarray:
    """The n-sample stretch of x [channels, samples] with the most energy,
    searched in steps of `step` samples."""
    if x.shape[-1] < n:
        raise ValueError(f"the backing is shorter than {n / fe.sr:.0f} s")
    c = np.concatenate([[0.0], np.cumsum((x.astype(np.float64) ** 2).sum(0))])
    starts = np.arange(0, x.shape[-1] - n + 1, step)
    a = int(starts[(c[starts + n] - c[starts]).argmax()])
    return x[:, a : a + n]


def mix_and_separate(
    voice: np.ndarray, backing: np.ndarray, vocal_db: float, device: torch.device
) -> np.ndarray:
    """voice [n] mixed into backing [2, n] at vocal_db relative to it (RMS),
    then Demucs's vocal stem of the mix: its mid, which is what inference
    feeds the decoder, back at the voice's level."""
    from inference.separation import separate_stems

    def rms(x):
        return float(np.sqrt((x**2).mean()))

    gain = 10.0 ** (vocal_db / 20.0) * rms(backing) / max(rms(voice), 1e-8)
    mix = backing + gain * voice[None]
    stems = separate_stems(torch.from_numpy(mix), fe.sr, device)
    return (stems["vocals"].mean(0).numpy() / gain).astype(np.float32)


def log_mel_l1(logmel, a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b)) // fe.hop_length * fe.hop_length
    ma, mb = (logmel(torch.from_numpy(x[:n])[None]) for x in (a, b))
    return float((ma - mb).abs()[..., 4:-4].mean())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument(
        "--voice",
        nargs="+",
        action="append",
        required=True,
        metavar="KIND NAME PATH",
        help="one voice: a label for the summary, a name, audio files or folders",
    )
    p.add_argument("--seconds", type=float, default=15.0, help="length of A and B")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=300.0,
        help="audio read per voice to pick A and B from",
    )
    p.add_argument("--shifts", type=float, nargs="*", default=[-6.0, 6.0])
    p.add_argument(
        "--backing",
        type=Path,
        help="an instrumental to mix each A into and separate it from again",
    )
    p.add_argument(
        "--vocal-db",
        type=float,
        default=0.0,
        help="level of the voice in that mix, relative to the backing (RMS)",
    )
    p.add_argument("--out", type=Path, default=Path("voice_check"))
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = p.parse_args()
    if len(args.voice) < 2 or any(len(v) < 3 for v in args.voice):
        p.error("give at least two --voice KIND NAME PATH...")

    from frontend.mel import LogMel
    from inference.resung import ResungShifter
    from inference.rubberband import default_formant_shift, rubberband_shift
    from speaker_similarity import SpeakerSimilarity

    dev = args.device
    if dev == "auto":
        dev = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(dev)
    shifter = ResungShifter(args.ckpt, device)
    score = SpeakerSimilarity("cpu")
    logmel = LogMel()
    rubberband = shutil.which("rubberband") is not None
    if not rubberband:
        print("no `rubberband` on PATH: the Rubber Band rows are skipped")
    args.out.mkdir(parents=True, exist_ok=True)

    def embed(x: np.ndarray) -> torch.Tensor:
        return score.embed(to_analysis_sr(x))

    win = round(args.seconds / F0_FRAME_S)
    n_win = round(args.seconds * fe.sr)
    backing = None
    if args.backing:
        x, sr = sf.read(args.backing, dtype="float32", always_2d=True)
        x = np.stack([resample(c, sr, fe.sr) for c in x.T])
        backing = loudest(np.broadcast_to(x, (2, x.shape[-1])), n_win)
    voices = []
    for kind, name, *paths in args.voice:
        x = load_voice([Path(s) for s in paths], args.max_seconds)
        voiced = shifter.track_f0(to_analysis_sr(x)) > 0
        starts = pick_windows(voiced, win)
        a, b = (x[round(s * F0_FRAME_S * fe.sr) :][:n_win] for s in starts)
        sung = [float(voiced[s : s + win].mean()) for s in starts]
        print(
            f"{name} ({kind}): {len(x) / fe.sr:.0f} s read, A at "
            f"{starts[0] * F0_FRAME_S:.0f} s ({sung[0]:.0%} sung), B at "
            f"{starts[1] * F0_FRAME_S:.0f} s ({sung[1]:.0%} sung)",
            flush=True,
        )
        voices.append(
            {
                "kind": kind,
                "name": name,
                "a": a,
                "emb_a": embed(a),
                "emb_b": embed(b),
                "timbre": shifter.timbre_of(a),
            }
        )

    rows = []
    for v in voices:
        others = [u for u in voices if u is not v]
        to_others = [float(v["emb_a"] @ u["emb_a"]) for u in others]
        donor = others[int(np.argmin(to_others))]
        a = v["a"]
        outs = {
            "original": a,
            "vocoder": shifter.copy_synthesis(a),
            "resung": shifter.shift(a, 0.0),
            "swap": shifter.shift(a, 0.0, timbre=donor["timbre"]),
        }
        for s in args.shifts:
            outs[f"resung{s:+g}"] = shifter.shift(a, s)
            if rubberband:
                outs[f"rubberband{s:+g}"] = rubberband_shift(
                    a, fe.sr, s, default_formant_shift(s)
                )
        if backing is not None:
            sep = mix_and_separate(a, backing, args.vocal_db, device)
            outs["separated"] = sep
            outs["resung_sep"] = shifter.shift(sep, 0.0)
            outs["resung_sep_content"] = shifter.shift(sep, 0.0, timbre=v["timbre"])
            outs["resung_sep_timbre"] = shifter.shift(
                a, 0.0, timbre=shifter.timbre_of(sep)
            )
        embs = {k: embed(y) for k, y in outs.items() if k != "original"}
        row = {
            "name": v["name"],
            "kind": v["kind"],
            "same": float(v["emb_a"] @ v["emb_b"]),
            "others": float(np.mean(to_others)),
            **{k: float(v["emb_a"] @ e) for k, e in embs.items()},
            "donor": donor["name"],
            "donor_own_timbre": float(donor["emb_a"] @ embs["resung"]),
            "donor_swapped": float(donor["emb_a"] @ embs["swap"]),
            "recon_l1": log_mel_l1(logmel, outs["vocoder"], outs["resung"]),
            "swap_l1": log_mel_l1(logmel, outs["resung"], outs["swap"]),
        }
        for k, y in outs.items():
            sf.write(args.out / f"{v['name']}_{k}.wav", y, fe.sr)
        rows.append(row)
        print({k: round(x, 3) if isinstance(x, float) else x for k, x in row.items()})

    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    systems = [k for k in rows[0] if k not in ("name", "kind", "donor")]
    width = max(len(k) for k in systems)
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    counts = ", ".join(f"{k}: {sum(r['kind'] == k for r in rows)}" for k in kinds)
    print(f"\nMeans per kind ({counts} voices)")
    print(" " * width + "".join(f"{k:>12}" for k in kinds))
    for s in systems:
        means = [np.mean([r[s] for r in rows if r["kind"] == k]) for k in kinds]
        print(f"{s:<{width}}" + "".join(f"{m:>12.3f}" for m in means))
    print(f"wavs and results.json in {args.out}/")


if __name__ == "__main__":
    main()
