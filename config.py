from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrontendConfig:
    # Resung's 44.1 kHz front end. The mel settings must match the
    # PC-NSF-HiFiGAN 2025.02 vocoder exactly (openvpi/SingingVocoders,
    # configs/ft_hifigan.yaml): natural-log magnitude mel, slaney filters.
    sr: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 40.0
    fmax: float = 16000.0
    log_mel_floor: float = 1e-5
    # ContentVec, RMVPE and ECAPA all run at 16 kHz.
    analysis_sr: int = 16000
    f0_hop: int = 160  # RMVPE frame: 10 ms at 16 kHz
    f0_threshold: float = 0.03  # RMVPE salience below this is unvoiced
    loudness_floor_db: float = -80.0
    target_lufs: float = -23.0
    # Same segmentation as the old cache, so the speaker-held-out split and
    # the evaluation clips match eval_final.json.
    segment_seconds: float = 5.12
    clip_frames: int = 224  # training crop: 224 * 512 samples = 2.60 s
    perturb_st: float = 6.0
    # Pinned model sources.
    content_repo: str = "lengyue233/content-vec-best"
    content_revision: str = "ab04aa7067b99ee05cc82499bc64916b980a1967"
    rmvpe_repo: str = "lj1995/VoiceConversionWebUI"
    rmvpe_file: str = "rmvpe.pt"
    rmvpe_revision: str = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
    timbre_repo: str = "speechbrain/spkrec-ecapa-voxceleb"
    timbre_dim: int = 192


# The predecessor's 22.05 kHz settings. Only the legacy WORLD + PitchUNet
# baseline (model.py, inference/pipeline.py, audio_preprocessing.py) uses them.
@dataclass
class AudioConfig:
    target_lufs: int = -23
    target_sr: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    win_length: int = 1024
    clip_seconds: float = 2.56
    segment_seconds: float = 5.12
    crepe_model_capacity: str = "full"
    voiced_threshold: float = 0.3
    fmin: int = 0
    fmax: int = 8000


@dataclass(frozen=True)
class DatasetSpec:
    hf_name: str
    subdir: str
    length_hr: int
    parquet_glob: str
    group_minutes: float = 5.0
    # Relative weight of this dataset when drawing training clips. 0 keeps
    # the dataset for validation and evaluation only.
    share: float = 1.0


@dataclass
class DataConfig:
    cache_dir: str = "./cache"
    val_split: float = 0.02

    datasets: dict[str, DatasetSpec] = field(
        default_factory=lambda: {
            "nsynth": DatasetSpec(
                hf_name="jg583/NSynth",
                subdir="nsynth",
                length_hr=2,
                parquet_glob="data/train/*.parquet",
            ),
            "vctk": DatasetSpec(
                hf_name="sanchit-gandhi/vctk",
                subdir="vctk",
                length_hr=10,
                parquet_glob="data/train-*.parquet",
            ),
            # This mirror is 16 kHz, so nothing above 8 kHz: fine as input
            # for the 16 kHz analysis models, wrong as a 44.1 kHz target.
            # Validation and evaluation only, which keeps the held-out clips
            # of eval_final.json.
            "opensinger_female": DatasetSpec(
                hf_name="CodecSR/opensinger_female",
                subdir="opensinger_fem",
                length_hr=3,
                parquet_glob="data/original-*.parquet",
                group_minutes=90,
                share=0.0,
            ),
            "opensinger_male": DatasetSpec(
                hf_name="CodecSR/opensinger_male",
                subdir="opensinger_male",
                length_hr=3,
                parquet_glob="data/original-*.parquet",
                group_minutes=90,
                share=0.0,
            ),
            # A tree of WAV files, not parquet: language/singer/technique/
            # song/group/NNNN.wav. 48 kHz, CC BY-NC-SA 4.0. Takes over
            # OpenSinger's training share: 60 minutes per singer, up from
            # 30. The held-out singer stays the same, since the split
            # depends on the singer set, not on the hours.
            "gtsinger": DatasetSpec(
                hf_name="GTSinger/GTSinger",
                subdir="gtsinger",
                length_hr=20,
                parquet_glob="**/*.wav",
                group_minutes=60,
                share=3.0,
            ),
        }
    )
    datasets_to_load: list[str] = field(
        default_factory=lambda: [
            "vctk",
            "opensinger_female",
            "opensinger_male",
            "gtsinger",
        ]
    )

    def __post_init__(self):
        assert self.val_split < 1
        unknown = set(self.datasets_to_load) - self.datasets.keys()
        assert not unknown, f"unknown datasets in datasets_to_load: {unknown}"


@dataclass
class TrainConfig:
    # Stage 1: the decoder against the frozen PC-NSF-HiFiGAN. Batch size is
    # per GPU; under torchrun the global batch is batch_size * world size.
    batch_size: int = 16
    num_workers: int = 6
    warmup_steps: int = 2000
    max_steps: int = 60000
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    betas = (0.8, 0.99)
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    # Share of clips whose content comes from the clean audio, as at
    # inference, rather than from the WORLD-shifted copy.
    p_clean_content: float = 0.2
    w_mel: float = 1.0
    # Waveform terms through the frozen vocoder, on a crop of wave_frames
    # frames from wave_batch clips: multi-resolution STFT and the vocoder's
    # own mel of the output. w_stft = w_wave_mel = 0 turns them off.
    w_stft: float = 1.0
    w_wave_mel: float = 1.0
    wave_batch: int = 8
    wave_frames: int = 64
    log_every: int = 100
    val_every: int = 2000
    val_batches: int = 50
    save_every: int = 2000
    keep_last: int = 5
    ckpt_dir: str = "./checkpoints"
    tb_dir: str = "./runs"


@dataclass(frozen=True)
class ModelConfig:
    # Resung's decoder: conv + self-attention blocks over the vocoder's frames,
    # with the timbre embedding entering every block through adaptive norms.
    content_dim: int = 768  # ContentVec layer 12
    content_bottleneck: int = 256
    dim: int = 384
    layers: int = 10
    heads: int = 6
    ff_mult: int = 4
    conv_kernel: int = 15
    cond_dim: int = 256
    f0_fourier: int = 16  # sin/cos pairs of log-F0
    dropout: float = 0.1


@dataclass(frozen=True)
class BaselineConfig:
    # The predecessor's WORLD + PitchUNet engine, kept as the baseline.
    vocoder_repo: str = "nvidia/bigvgan_base_22khz_80band"
    perturb_st: int = 6


@dataclass
class PitchUNetConfig:
    base_channels: int = 64
    channel_mults = [1, 2, 4, 6]  # 64, 128, 256, 384
    shift_emb_dim: int = 64
    f0_emb_dim: int = 64
    attention_in_bottleneck: bool = True
