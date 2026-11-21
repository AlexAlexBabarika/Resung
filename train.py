"""Stage 1: train the decoder against the frozen PC-NSF-HiFiGAN.

    uv run train.py [--max-steps N] [--batch-size B] ...                # one device
    uv run torchrun --nproc_per_node=4 train.py ...                     # one per GPU

Per step: ContentVec reads the WORLD-shifted 16 kHz clip (or, for a share of
clips, the clean one), the decoder predicts the mel from that content plus the
clean clip's F0, loudness and timbre, and the loss compares it with the clean
clip's mel, directly and after the frozen vocoder.

The run reports its measured speed and what the full run would take
(--price-per-hour turns that into dollars).
"""

import argparse
import math
import os
import re
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.writer import SummaryWriter

from config import DataConfig, FrontendConfig, ModelConfig, TrainConfig
from data import ResungDataset
from frontend.content import ContentVec
from frontend.mel import LogMel
from losses import WaveLoss, masked_l1
from model import ResungDecoder, count_params
from vocoder import checkpoint_path, load_generator

load_dotenv()  # HF_TOKEN, for authenticated model downloads
# PyTorch's own MPS stft emits a resize deprecation warning on every call.
warnings.filterwarnings(
    "ignore", message="An output with one or more elements was resized"
)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.step = 0
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        # Ramped so early averages aren't dominated by the random init.
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])


class Stage1(nn.Module):
    """Everything a training step needs; only the decoder is trainable."""

    def __init__(
        self,
        cfg: TrainConfig,
        device: torch.device,
        model_cfg: ModelConfig = ModelConfig(),
        bf16: bool | None = None,
    ):
        """bf16: autocast the decoder and ContentVec to bfloat16 (default: on
        CUDA only; the vocoder losses always run in float32)."""
        super().__init__()
        self.cfg = cfg
        self.bf16 = device.type == "cuda" if bf16 is None else bf16
        self.decoder = ResungDecoder(model_cfg)
        self.content = ContentVec()
        self.logmel = LogMel()
        self.wave = WaveLoss(load_generator("cpu"), cfg.wave_frames, cfg.wave_batch)
        self.to(device)

    def losses(
        self,
        batch: dict[str, torch.Tensor],
        decoder: nn.Module | None = None,
        clean_content: bool | None = None,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """decoder: the module to run (the DDP wrapper in training, the EMA
        copy in validation). clean_content: force clean (True) or perturbed
        (False) content for every clip; None mixes by p_clean_content.
        The parts come back as detached tensors: converting them to floats
        every step would stall the GPU."""
        if decoder is None:
            decoder = self.decoder
        dev = next(self.decoder.parameters()).device
        b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        mask = b["mask"]
        with torch.no_grad():
            mel_tgt = self.logmel(b["audio"])
            if clean_content is None:
                pick = torch.rand(mask.shape[0], device=dev) < self.cfg.p_clean_content
            else:
                pick = torch.full((mask.shape[0],), clean_content, device=dev)
            content_audio = torch.where(pick[:, None], b["audio16"], b["audio16_pert"])
            with torch.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=self.bf16
            ):
                content = self.content(content_audio).float()
        with torch.autocast(
            device_type=dev.type, dtype=torch.bfloat16, enabled=self.bf16
        ):
            pred = decoder(content, b["f0"], b["uv"], b["loudness"], b["timbre"], mask)
        pred = pred.float()
        mel = masked_l1(pred, mel_tgt, mask)
        loss = self.cfg.w_mel * mel
        parts = {"mel": mel.detach()}
        if self.cfg.w_stft > 0 or self.cfg.w_wave_mel > 0:
            wave = self.wave(pred, b["f0"], b["audio"], mask, rng)
            if wave is not None:
                stft, wave_mel = wave
                loss = loss + self.cfg.w_stft * stft + self.cfg.w_wave_mel * wave_mel
                parts |= {"stft": stft.detach(), "wave_mel": wave_mel.detach()}
        parts["loss"] = loss.detach()
        return loss, parts


def lr_lambda(cfg: TrainConfig):
    def f(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return f


def setup_distributed(requested: str) -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available() and requested in ("auto", "cuda"):
            device = torch.device("cuda", local)
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", device_id=device)
            return dist.get_rank(), world, device
        dist.init_process_group("gloo")
        return dist.get_rank(), world, torch.device("cpu")
    if requested != "auto":
        return 0, 1, torch.device(requested)
    if torch.cuda.is_available():
        return 0, 1, torch.device("cuda")
    if torch.backends.mps.is_available():
        return 0, 1, torch.device("mps")
    return 0, 1, torch.device("cpu")


_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def step_ckpts(ckpt_dir: Path) -> list[tuple[int, Path]]:
    found = [
        (int(m.group(1)), p)
        for p in ckpt_dir.glob("step_*.pt")
        if (m := _STEP_RE.match(p.name))
    ]
    return sorted(found)


def save_ckpt(
    path: Path, stage: Stage1, ema: EMA, opt, sched, step: int, best_val: float
):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "decoder": stage.decoder.state_dict(),
            "ema": {"shadow": ema.shadow, "step": ema.step},
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": step,
            "best_val": best_val,
            "model_config": asdict(stage.decoder.cfg),
            "frontend_config": asdict(FrontendConfig()),
            "train_config": asdict(stage.cfg),
        },
        tmp,
    )
    os.replace(tmp, path)


@torch.no_grad()
def validate(
    stage: Stage1, ema: EMA, eval_decoder: nn.Module, loader, step: int, writer
) -> float:
    """EMA weights, fixed crop positions. Reports the mel L1 with clean
    content (the unshifted inference case) and with perturbed content (pitch
    from the F0 track only); their mean picks best.pt."""
    ema.copy_to(eval_decoder)
    eval_decoder.eval()
    sums: dict[str, float] = {}
    n = 0
    rng = torch.Generator().manual_seed(0)
    for i, batch in enumerate(loader):
        if i >= stage.cfg.val_batches:
            break
        for name, clean in (("clean", True), ("pert", False)):
            _, parts = stage.losses(batch, eval_decoder, clean_content=clean, rng=rng)
            for k, v in parts.items():
                sums[f"{k}_{name}"] = sums.get(f"{k}_{name}", 0.0) + v.item()
        n += 1
    if n == 0:
        return math.inf
    means = {k: v / n for k, v in sums.items()}
    for k, v in means.items():
        writer.add_scalar(f"val/{k}", v, step)
    score = 0.5 * (means["mel_clean"] + means["mel_pert"])
    print(
        f"\n[val step {step}] mel L1 clean {means['mel_clean']:.4f}, "
        f"perturbed {means['mel_pert']:.4f}"
        + (f", stft {means['stft_pert']:.4f}" if "stft_pert" in means else "")
    )
    return score


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Resung stage 1: decoder vs the frozen vocoder."
    )
    for name in (
        "max_steps",
        "val_every",
        "save_every",
        "batch_size",
        "num_workers",
        "log_every",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("stage1"),
        help="best.pt, final.pt, checkpoints/ and runs/ go here (the repo root keeps the baseline's)",
    )
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument(
        "--datasets", nargs="+", help="subset of DataConfig.datasets_to_load"
    )
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--price-per-hour", type=float, default=None, help="for the cost projection"
    )
    p.add_argument(
        "--project-steps",
        type=int,
        default=None,
        help="steps to project the run time for (default: TrainConfig.max_steps)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = TrainConfig()
    for name in (
        "max_steps",
        "val_every",
        "save_every",
        "batch_size",
        "num_workers",
        "log_every",
    ):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    project_steps = args.project_steps or TrainConfig().max_steps
    rank, world, device = setup_distributed(args.device)
    main_proc = rank == 0
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        # Fixed shapes every step: let cuDNN pick its fastest kernels, and
        # let the float32 vocoder pass use TF32 matmuls.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    out_dir = args.out_dir
    ckpt_dir = out_dir / cfg.ckpt_dir
    if (out_dir / "final.pt").exists():
        if main_proc:
            print(
                f"{out_dir / 'final.pt'} exists: training is complete. Delete it to start a new run."
            )
        sys.exit(0)
    # One rank downloads the pretrained models; the others wait for it.
    if main_proc:
        checkpoint_path()
        fe = FrontendConfig()
        snapshot_download(fe.content_repo, revision=fe.content_revision)
    if world > 1:
        dist.barrier()

    stage = Stage1(cfg, device)
    eval_decoder = ResungDecoder(stage.decoder.cfg).to(device)
    if main_proc:
        print(
            f"decoder: {count_params(stage.decoder) / 1e6:.2f} M params, device {device}, world {world}"
        )
    decoder: nn.Module = stage.decoder
    if world > 1:
        # The decoder's only buffer is a constant, so there is nothing to
        # broadcast before every forward.
        decoder = DDP(
            stage.decoder,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    opt = torch.optim.AdamW(
        stage.decoder.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    ema = EMA(stage.decoder, cfg.ema_decay)

    step, best_val = 0, math.inf
    ckpts = step_ckpts(ckpt_dir) if ckpt_dir.exists() else []
    if ckpts:
        ck = torch.load(ckpts[-1][1], map_location=device, weights_only=False)
        stage.decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        ema.shadow, ema.step = ck["ema"]["shadow"], ck["ema"]["step"]
        step, best_val = ck["step"], ck["best_val"]
        if main_proc:
            print(f"resumed from {ckpts[-1][1]} (step {step})")

    data_kwargs = {}
    if args.cache_dir:
        data_kwargs["cache_dir"] = args.cache_dir
    if args.datasets:
        data_kwargs["dataset_keys"] = args.datasets
    train_loader = ResungDataset.make_loader(
        "train",
        cfg.batch_size,
        cfg.num_workers,
        rank=rank,
        world_size=world,
        seed=args.seed + step,
        **data_kwargs,
    )
    val_loader = (
        ResungDataset.make_loader("val", cfg.batch_size, cfg.num_workers, **data_kwargs)
        if main_proc
        else None
    )
    writer = SummaryWriter(log_dir=str(out_dir / cfg.tb_dir)) if main_proc else None
    if main_proc:
        assert val_loader is not None
        print(
            f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val segments "
            f"({', '.join(args.datasets or DataConfig().datasets_to_load)}), "
            f"global batch {cfg.batch_size * world}, "
            f"{cfg.max_steps} steps"
        )

    timer_start, timer_step = None, None
    decoder.train()
    while step < cfg.max_steps:
        for batch in train_loader:
            loss, parts = stage.losses(batch, decoder)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                stage.decoder.parameters(), cfg.grad_clip
            )
            opt.step()
            sched.step()
            ema.update(stage.decoder)
            step += 1

            # Speed, measured after a warm-up of 10 steps.
            if step == 10:
                timer_start, timer_step = time.perf_counter(), step
            report = (
                main_proc
                and timer_start is not None
                and step > timer_step
                and (step in (60, 1000) or step == cfg.max_steps)
            )
            if report:
                its = (step - timer_step) / (time.perf_counter() - timer_start)
                hours = project_steps / its / 3600
                cost = (
                    f", about ${hours * args.price_per_hour:.0f}"
                    if args.price_per_hour
                    else ""
                )
                print(
                    f"\n[speed] {its:.2f} it/s at global batch {cfg.batch_size * world}: "
                    f"{project_steps} steps would take {hours:.1f} h{cost} (training only)"
                )
                assert writer is not None
                writer.add_scalar("speed/it_per_s", its, step)

            if main_proc and step % cfg.log_every == 0:
                assert writer is not None
                lr = sched.get_last_lr()[0]
                parts_f = {k: v.item() for k, v in parts.items()}
                for k, v in parts_f.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                msg = " ".join(f"{k} {v:.4f}" for k, v in parts_f.items())
                print(f"step {step} {msg} lr {lr:.2e}", flush=True)

            if step % cfg.val_every == 0 or step == cfg.max_steps:
                if main_proc:
                    assert val_loader is not None and writer is not None
                    score = validate(stage, ema, eval_decoder, val_loader, step, writer)
                    if score < best_val:
                        best_val = score
                        save_ckpt(
                            out_dir / "best.pt", stage, ema, opt, sched, step, best_val
                        )
                decoder.train()
                if world > 1:
                    dist.barrier()

            if main_proc and step % cfg.save_every == 0:
                save_ckpt(
                    ckpt_dir / f"step_{step}.pt", stage, ema, opt, sched, step, best_val
                )
                for _, old in step_ckpts(ckpt_dir)[: -cfg.keep_last]:
                    old.unlink(missing_ok=True)

            if step >= cfg.max_steps:
                break

    if main_proc:
        save_ckpt(out_dir / "final.pt", stage, ema, opt, sched, step, best_val)
        assert writer is not None
        writer.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
