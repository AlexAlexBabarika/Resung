# Resung

Song pitch shifting that aims to sound as if the singer re-sang the song in the
new key and the band retuned their instruments.

**Status: stopped after stage 1, as a negative result (November 2026).** Resung
was built to succeed
[SelfSupervised-Pitch-Shifter](https://github.com/AlexAlexBabarika/SelfSupervised-Pitch-Shifter),
which moved the pitch with WORLD and cleaned the result with a U-Net. Listening
tests traced that model's robotic sound to WORLD's pitch change itself, so Resung
replaces the WORLD front end on the vocals: a frozen ContentVec encoder captures
what is sung, RMVPE tracks the pitch, and a decoder with an NSF-HiFiGAN vocoder
generates the voice at the new pitch. Training is self-supervised. Instruments
stay on Signalsmith Stretch and drums stay untouched.

Stage 1 moved the pitch of a real song accurately. But on vocals from commercial
songs the output no longer sounds like the singer, and the vocoder alone already
sounds robotic on them. [What stage 1 showed](#what-stage-1-showed) has the
measurements. The planned adversarial stage, which would also have fine-tuned the
vocoder, was not built.

Built: the 44.1 kHz front end and training cache (`frontend/`, `data.py`,
`Process_HF_Datasets.py`), the decoder (`model.py`), stage-1 training against the
frozen vocoder (`train.py`), the vocal engine (`inference/resung.py`) and the
checks below. Resung was never scored in `evaluate.py` or in a blind test, so
inference runs on the inherited pipeline described further down.

## Front end

Every training clip carries, from the unshifted vocal: the 44.1 kHz audio (the
target), F0 from [RMVPE](https://github.com/Dream-High/RMVPE), A-weighted loudness
on the vocoder's frames, and an ECAPA-TDNN speaker embedding taken from another
segment by the same voice. The content encoder,
[ContentVec](https://github.com/auspicious3000/contentvec), runs on the GPU during
training on a copy of the clip that WORLD has shifted by a random amount of up to
6 semitones either way. The pitch in the content features is then wrong on
purpose, and the decoder has to
take its pitch from the F0 track. The mel settings match the PC-NSF-HiFiGAN
vocoder (44.1 kHz, 128 bands, hop 512, 40 Hz–16 kHz).

## Decoder and stage 1

The decoder (`model.py`, 32 M parameters) turns those inputs into the vocoder's
mel: the content passes through a 256-wide bottleneck and is interpolated onto
the mel frames, joined by Fourier features of log-F0 and the loudness, and run
through ten conv + self-attention blocks (rotary positions); the timbre
embedding enters every block through adaptive layer norms whose residual gates
start at zero. [PC-NSF-HiFiGAN](https://github.com/openvpi/SingingVocoders)
(OpenVPI, weights CC BY-NC-SA 4.0, downloaded from the official release and
checksum-verified; generator code vendored in `vocoder/`, MIT) renders the mel
at the F0 it is given.

Stage 1 trains the decoder against the frozen vocoder: mel L1, plus a
multi-resolution STFT loss and the vocoder-mel L1 on a crop rendered through the
vocoder. A fifth of the clips use clean content, as inference does.
`scripts/vocoder_check.py` checks the vocoder on cached clips (alignment, copy
synthesis, its own pitch shift); `scripts/listen_stage1.py` renders a trained
checkpoint's reconstructions and shifts next to the vocoder's own shift.

`scripts/voice_check.py` measures whether the output still sounds like the
singer, with a speaker-verification model the decoder was not trained on
(`speaker_similarity.py`, WavLM-base-plus-sv). It takes audio files, so it runs
without the training cache: per voice it scores the vocoder alone, the decoder
at the original pitch, the decoder with another voice's timbre embedding, and
shifts next to Rubber Band, against the same voice on other audio and against
other voices. `--backing` mixes each clean voice into an instrumental and
separates it again first.

## What stage 1 showed

Stage 1 trained to step 46,900 of 60,000 and was stopped when the held-out loss
levelled off. Everything below uses the step-46,000 checkpoint.

**Pitch.** On one commercial song, shifted by +6, -2 and -6 semitones, the median
pitch error of the shifted vocal was 3.5, 3.7 and 4.1 cents, with about 95% of
frames within 50 cents.

**Voice.** `scripts/voice_check.py`, on 15-second excerpts, decoder at the
original pitch. Scores are speaker-similarity cosines to the original excerpt.
For scale, the same singer on other audio scores about 0.93 and a different
singer 0.60 to 0.73.

| Voices | Vocoder alone | Decoder |
|---|---|---|
| Clean, trained on (4 GTSinger singers) | 0.99 | 0.96 |
| Clean, never heard (1 GTSinger and 6 OpenSinger singers) | 0.98 | 0.97 |
| Vocals separated from 3 commercial songs, 12 excerpts | 0.94 | 0.82 (0.62 to 0.92) |

- The voice is lost before any shift: on the song vocals the drop is there at the
  original pitch.
- Clean voices keep their identity, including singers the decoder never heard.
  Shifted by 6 semitones they score 0.91 to 0.93 with Resung, against 0.79 to 0.88
  with Rubber Band.
- The decoder uses the timbre embedding. With another clean voice's embedding,
  similarity to the original falls from 0.96 to 0.81 and similarity to the donor
  rises from 0.49 to 0.70.
- Simple processing of a clean voice does not reproduce the loss. Mixing it into
  an instrumental and separating it again with Demucs, a plate reverb, a double
  and a harmony each cost 0.05 or less. What it is about produced vocals is not
  established.

**By ear** (one listener, not blind, unshifted excerpts of the three songs). The
original voice could not be recognised in the decoder's output. The vocoder's
output kept the voice more or less, but it broke up and sounded robotic, like a
badly made auto-tune effect. The similarity score gave those vocoder outputs 0.82
to 0.99, so it does not register this and cannot be the only measure.

**Pitch track.** On one of the songs RMVPE's track jumped by more than 4
semitones 15 to 64 times in a 15-second excerpt, against 0 or 1 on clean voices.
A continuous path through RMVPE's salience removed the jumps, but by ear it
changed the voice and the robotic sound stayed, so it was not adopted. This was
measured with throwaway scripts that are not in the repository.

**Why it stopped here.** The vocoder alone is the best the decoder can do, and on
produced vocals it already fails by ear. More decoder training or data would not
get past that. Whether fine-tuning the vocoder in the adversarial stage would
have helped was not tested.

Limits: one listener; three songs; one scoring model; the song excerpts are
commercial recordings and are not in the repository, so only the clean-voice rows
can be reproduced from it.

## Current pipeline (inherited)

A pitch shifter for songs. It separates stems with Demucs and, by default,
transposes the whole song: the vocals with formant control, bass and other
instruments with [Signalsmith Stretch](https://signalsmith-audio.co.uk/code/stretch/),
drums untouched. `--mode vocals` shifts only the vocals against the original
backing. There are two engines for the vocals, and by default (`--engine auto`)
the shift size picks one: Rubber Band up to 3 semitones, the model above that
(Rubber Band if no checkpoint is found).

- **`rubberband`**: [Rubber Band](https://breakfastquay.com/rubberband/)
  with control over how far the formants follow the pitch. In `evaluate.py` it is
  ahead of the model on every measure at 1 semitone and on the round trip at 3.
- **`model`**: a WORLD shift moves the pitch, a conditional U-Net predicts the
  clean mel-spectrogram of the shifted signal, and a BigVGAN vocoder resynthesizes
  the audio. It needs a trained checkpoint
  ([weights on Hugging Face](https://huggingface.co/AlexAlexBabarika/SelfSupervised-Pitch-Shifter)).
  At 6 semitones it was preferred over Rubber Band in a small blind listening test,
  but large shifts still sound somewhat robotic. It also scores ahead at 6 semitones
  on the envelope and round-trip measures, but mostly in a band the test clips can't
  give Rubber Band; below 6 kHz Rubber Band is ahead.

### How the model engine works

```
song.wav
  └─ Demucs (htdemucs) ── vocals ──► mid ──► [resample 22.05 kHz]
                       │          │                 │
                       │          │          F0 (CREPE)
                       │          │                 │
                       │          │          WORLD shift ──► log-mel
                       │          │                 │
                       │          │          PitchUNet (mel → mel)
                       │          │                 │
                       │          │          BigVGAN (mel → audio)
                       │          │                 │
                       │          │          + band above 8 kHz from a
                       │          │            44.1 kHz WORLD shift
                       │          │                 │
                       │          └─ side ─► Rubber Band shift
                       │                            │
                       └─ drums/bass/other ─────────┴──► remix ──► shifted.wav
```

The core model, **`PitchUNet`** (`pitch_unet.py`), is a 2-D U-Net over log-mel
spectrograms. It is conditioned on:

- **Shift amount** (semitones), embedded and injected through FiLM layers in every
  residual block. A `cond_mask` distinguishes "no conditioning" from "shift = 0",
  enabling classifier-free-style conditioning dropout during training.
- **F0 contour** (`log2(f0+1)` + voiced mask), extracted with
  [`torchcrepe`](https://github.com/maxrmorrison/torchcrepe). F0 features are
  concatenated at the input and also drive **gated skip connections**, so the
  decoder leans on the pitch track when reconstructing harmonics.
- **Harmonic template**: the F0 track rendered as a picture of where its harmonics
  fall on the mel axis, concatenated at the input.

A self-attention block sits in the bottleneck. The network predicts a residual on
its input mel and the output head is zero-initialized, so training starts from the
identity (i.e. from the quality of the WORLD shift).

Its training (in the predecessor's repository) used an L1 + multi-resolution + difference loss on the mel and
a multi-resolution STFT loss on audio rendered through the frozen vocoder
(set `w_wave = 0` to turn the latter off), AdamW
with cosine decay and warmup, EMA weights, gradient clipping, and bf16 autocast on
CUDA. The input mel comes from a WORLD round trip (shift by `+s`, then back by
`-s`) of the original clip, so the network learns to remove the artifacts of a
WORLD shift while the target stays the untouched original. Inference applies a
single WORLD shift and hands the result to the network. WORLD models one voice,
so backing vocals and heavy reverb in the input degrade the result.

At inference F0 and WORLD run once over the whole signal, the U-Net runs over
overlapping windows whose predicted mels are cross-faded, and the vocoder makes a
single pass over the stitched mel. For stereo vocals the mid channel goes through
the model and Rubber Band shifts the side channel with the same formant setting.
On a downward shift the WORLD envelope is moved by a quarter of the shift before
the U-Net sees it, so the formants follow a little, as a real voice's do.

## Layout

| Path | Purpose |
|------|---------|
| `config.py` | All hyperparameters (`FrontendConfig`, `DataConfig`, `TrainConfig`, `ModelConfig`; `AudioConfig` is the baseline's) |
| `Download_HF_Datasets.py` | Pull dataset parquet shards from the Hugging Face Hub |
| `Process_HF_Datasets.py` | Stream the sources → the 44.1 kHz `.npz` cache + index files |
| `frontend/` | Resung's front end: vocoder mel, loudness, RMVPE (vendored, MIT), ContentVec, ECAPA, WORLD perturbation, cache writer |
| `data.py` | `ResungDataset` / dataloader: crops, F0 on the vocoder's frames, the perturbed content input |
| `scripts/check_cache.py` | Sanity checks on the cache before renting a GPU |
| `tests/` | `uv run pytest` (`-m 'not models'` skips the pretrained-model tests) |
| `audio_preprocessing.py` | Baseline only: CREPE F0, WORLD shift, 22.05 kHz mels |
| `model.py` | `ResungDecoder`: content, F0, loudness and timbre → the vocoder's mel |
| `vocoder/` | PC-NSF-HiFiGAN generator (vendored, MIT) and the checksum-verified weight download |
| `losses.py` | Mel L1, and the STFT and vocoder-mel losses through the frozen vocoder |
| `train.py` | Stage 1: one device or `torchrun` across GPUs, EMA, checkpoints, TensorBoard, speed and cost projection |
| `scripts/vocoder_check.py`, `scripts/listen_stage1.py` | Vocoder check; listening and pitch check of a trained checkpoint |
| `speaker_similarity.py` | Speaker similarity for evaluation (WavLM-base-plus-sv, not the conditioning model) |
| `scripts/voice_check.py` | Does the output keep the singer? Similarity per system on any audio files |
| `scripts/vast.sh` | Rented-GPU workflow: setup, data, smoke, train, listen |
| `pitch_unet.py` | Baseline only: the predecessor's `PitchUNet` |
| `inference.py` | CLI entrypoint: separate → shift → remix |
| `inference/` | Demucs separation, vocoders (BigVGAN, HiFi-GAN), stitching, pipeline |

## Setup

Requires Python ≥ 3.13. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

To download datasets you need a Hugging Face token:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=...
```

## Inference

Shift a song down two semitones (separates stems, shifts vocals, remixes):

```bash
uv run inference.py path/to/song.wav -s -2
```

Both engines need the `rubberband` command-line tool (the model engine uses it for
the stereo side of the vocal)
(`brew install rubberband` on macOS, `apt install rubberband-cli` on Debian/Ubuntu).

Useful flags:

- `-s, --semitones` — shift amount; positive = up, negative = down (required)
- `-o, --output` — output path (default: `<input>_shifted_<+N>st.wav` next to input)
- `--engine` — `auto` (default: Rubber Band up to 3 semitones, the model above),
  `rubberband` or `model`
- `--mode` — `song` (default) transposes vocals, bass and other instruments and
  leaves the drums alone; `vocals` shifts only the vocals
- `--formant-shift` — semitones the formants move; `0` is full formant
  preservation. Default: with Rubber Band they follow a quarter of the shift, as a
  real voice does; with the model they follow a quarter of a downward shift and
  stay put on an upward one, which won a small blind listening test
- `--no-separate` — skip Demucs and shift the input as it is; meant for solo vocal
  recordings (the `model` engine only handles a single voice)

Flags for the `model` engine:

- `--ckpt` — PitchUNet checkpoint, a training `.pt` or the released
  `model.safetensors` (default: `./best.pt`, else `./final.pt`, else latest
  `./checkpoints/step_*.pt`)
- `--no-ema` — use raw weights instead of the EMA shadow
- `--vocoder-repo` — HF repo of the vocoder; BigVGAN and HiFi-GAN repos both work
  (default: `BaselineConfig.vocoder_repo`, the one the model was trained against)
- `--device` — `auto` | `cuda` | `mps` | `cpu`
- `-v, --verbose` — progress output

When separating, the individual stems are also written to a `*_stems/` directory
next to the output.

## Training

```bash
# 1. download dataset shards into ./datasets
uv run Download_HF_Datasets.py

# 2. preprocess into ./cache/<dataset>/*.npz (+ index json, timbre table)
uv run Process_HF_Datasets.py
uv run scripts/check_cache.py

# 3. stage 1, on one device or one process per GPU
uv run train.py
uv run torchrun --standalone --nproc_per_node=4 train.py
```

`train.py` takes `--max-steps`, `--val-every`, `--save-every`, `--batch-size` (per
GPU), `--num-workers` (per GPU), `--log-every`, `--out-dir`, `--cache-dir` and
`--datasets` to override `TrainConfig` for a run, and `--price-per-hour` to turn
its speed report into a cost. It writes to `stage1/` by default and resumes from
the latest checkpoint in `<out-dir>/checkpoints/`.

### On a rented GPU (Vast.ai)

Pick an Ubuntu image with an NVIDIA driver, one or more 24 GB+ cards, about 16
CPU cores per GPU (the loader runs WORLD on the CPU for every clip), 128 GB of RAM
for a multi-GPU box, and about 150 GB of disk. Clone the repository, then:

```bash
HF_TOKEN=hf_... bash scripts/vast.sh setup   # system packages, uv, Python deps
bash scripts/vast.sh data                    # download, preprocess, sanity-check the cache
PRICE=2.6 bash scripts/vast.sh smoke         # vocoder check + 60 steps; projects hours and $
tmux new -s train 'PRICE=2.6 bash scripts/vast.sh train'
bash scripts/vast.sh listen                  # reconstructions and shifts from stage1/best.pt
```

Stage 1 writes into `stage1/` (the repository root keeps the baseline's
`best.pt`). Copy `stage1/best.pt` off the machine before destroying the instance.

Datasets, the per-dataset hour budget, and the train/val split are configured in
`DataConfig`. Validation holds out whole speakers, singers and instruments, so
the split is at least `val_split` of each dataset rather than exactly that. Out of
the box it trains on VCTK speech (48 kHz, up to 10 h, 5 minutes per speaker) and
GTSinger (48 kHz, 20 singers in 9 languages, 60 minutes each; CC BY-NC-SA 4.0, so
non-commercial use only), drawn by the `share` in each `DatasetSpec`: a quarter
speech, three quarters singing. OpenSinger (male/female) is processed for
validation only: the Hugging Face copy is 16 kHz, which is fine as input to the
16 kHz analysis models but has nothing above 8 kHz to serve as a 44.1 kHz target.
NSynth is still defined but no longer loaded. Checkpoints land in `stage1/checkpoints/`
(pruned to the last few), with `best.pt` written whenever validation improves and
`final.pt` at the end of training; logs go to `stage1/runs/` (TensorBoard).

## Evaluation

`evaluate.py` still reads the predecessor's 22.05 kHz cache. It was not ported to
the new one, so it scores the baseline only:

```bash
uv run evaluate.py            # uses ./best.pt (the baseline) and the held-out validation clips
```

Scores the full system against `librosa`, Rubber Band (if the `rubberband` CLI is
installed), PSOLA (Praat), WORLD alone, WORLD + vocoder (no U-Net) and the vocoder
alone, per shift size. A single shift has no ground truth, so it is
scored on pitch accuracy (CREPE F0 against the source F0 scaled by the shift), on
how many voiced frames survive, and on how far the spectral envelope moved from the
source. A round trip (`+s` then `-s`) is compared with the original clip by log-mel
L1 and mel-cepstral distortion. Results are printed as a table and written to
`eval_results.json`.

## Notes

- The baseline model runs at 22.05 kHz mono with 80-bin mels (`n_fft=1024`,
  `hop=256`, up to 8 kHz); Demucs operates at its native 44.1 kHz stereo, and
  inference output is 44.1 kHz. Resung's cache is 44.1 kHz.
- The cache holds segments of up to 5.12 s, loudness-normalised to -23 LUFS; each
  training step crops a 2.6 s clip (224 vocoder frames) at a random offset.
- The default vocoder is BigVGAN base (22 kHz, 80 bands); its code is vendored in
  `inference/bigvgan/` (MIT, NVIDIA). The HiFi-GAN UNIVERSAL_V1 generator in
  `inference/hifigan/` is still supported. Weights are fetched from the Hub.
