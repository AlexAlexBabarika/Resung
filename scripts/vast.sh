#!/usr/bin/env bash
# Train Resung (stage 1) on a rented GPU box with one or more GPUs (written
# for Vast.ai, works on any Ubuntu machine with an NVIDIA driver). Run from
# the repository root:
#
#   HF_TOKEN=hf_... bash scripts/vast.sh setup   # system packages, uv, Python deps
#   bash scripts/vast.sh data                    # download + preprocess + sanity check
#   PRICE=2.6 bash scripts/vast.sh smoke         # vocoder check + 60 steps; projects
#                                                # the full run's hours (and $ at PRICE/h)
#   PRICE=2.6 bash scripts/vast.sh train         # the real run; resumes if interrupted
#   bash scripts/vast.sh listen                  # step-4 check on stage1/best.pt
#
# Run `train` inside tmux so it survives a dropped SSH connection. Every GPU
# gets one training process (torchrun) and its own share of the CPU cores.
# If multi-GPU NCCL hangs at start-up, retry with NCCL_P2P_DISABLE=1.
set -euo pipefail
cd "$(dirname "$0")/.."

# CPUs this container may use. Vast.ai limits a container with a cgroup CPU
# quota, but nproc still counts every core on the host, often 64 to 256. One
# data-loader worker per host core, each starting an OpenMP pool of that many
# threads, runs out of threads ("libgomp: Thread creation failed").
cpus_available() {
    local n quota period
    n="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    if [ -r /sys/fs/cgroup/cpu.max ]; then                 # cgroup v2
        read -r quota period < /sys/fs/cgroup/cpu.max
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then  # cgroup v1
        quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
        period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
    fi
    # "max" (v2) or -1 (v1) means no quota.
    if [[ "${quota:-}" =~ ^[0-9]+$ ]] && [ "$quota" -gt 0 ]; then
        local limit=$(( (quota + period - 1) / period ))
        [ "$limit" -lt "$n" ] && n="$limit"
    fi
    echo "$n"
}

gpus_available() {
    if command -v nvidia-smi >/dev/null; then nvidia-smi -L | grep -c '^GPU' ; else echo 1; fi
}

CPUS="$(cpus_available)"
GPUS="${GPUS:-$(gpus_available)}"
# Loader workers per training process; the cores left over run the
# training processes themselves.
WORKERS="${WORKERS:-$(( (CPUS - GPUS) / GPUS > 1 ? (CPUS - GPUS) / GPUS : 1 ))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPUS / GPUS > 1 ? CPUS / GPUS : 1 ))}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

setup() {
    $SUDO apt-get update
    # build-essential: pyworld compiles from source. rubberband-cli: only for
    # the Rubber Band row of evaluate.py.
    $SUDO apt-get install -y build-essential git tmux curl
    $SUDO apt-get install -y rubberband-cli || echo "rubberband-cli not available; evaluate.py will skip that row"
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv sync
    if [ -n "${HF_TOKEN:-}" ] && [ ! -f .env ]; then
        (umask 077; printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > .env)
    fi
    uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device visible'; print('GPU:', torch.cuda.get_device_name(0))"
}

data() {
    uv run Download_HF_Datasets.py
    uv run Process_HF_Datasets.py
    uv run scripts/check_cache.py
}

free_space() {
    echo "free disk space here: $(df -h . | awk 'NR==2 {print $4}')"
    echo "CPUs: $CPUS usable (nproc says $(nproc 2>/dev/null || sysctl -n hw.ncpu)), $GPUS GPUs, $WORKERS loader workers per GPU"
}

run_train() {
    uv run torchrun --standalone --nproc_per_node="$GPUS" train.py \
        --num-workers "$WORKERS" ${PRICE:+--price-per-hour "$PRICE"} "$@"
}

smoke() {
    # Kept next to the repository: /tmp sits on the container's small root
    # disk on Vast.ai, and each checkpoint is about 500 MB.
    free_space
    rm -rf .smoke
    uv run scripts/vocoder_check.py --n 4 --out .smoke/vocoder_check
    run_train --max-steps 60 --val-every 30 --save-every 30 --log-every 10 --out-dir .smoke
    rm -rf .smoke
    echo "smoke test passed"
}

train() {
    free_space
    run_train 2>&1 | tee -a train.log
}

listen() {
    uv run scripts/listen_stage1.py --ckpt stage1/best.pt --out listen_stage1 2>&1 | tee listen_stage1.log
}

case "${1:-}" in
    setup) setup ;;
    data) data ;;
    smoke) smoke ;;
    train) train ;;
    listen) listen ;;
    eval) echo "evaluate.py is being ported to the new cache (step 7)"; exit 1 ;;
    *) sed -n '2,16p' "$0"; exit 1 ;;
esac
