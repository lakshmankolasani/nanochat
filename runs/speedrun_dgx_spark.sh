#!/bin/bash

# DGX Spark (GB10) adaptation of nanochat speedrun
# Based on community-verified settings from:
#   - https://forums.developer.nvidia.com/t/anyone-got-nanochat-training-working-on-the-dgx-spark/348537
#   - https://github.com/karpathy/nanochat/pull/475 (Blackwell SM100 SDPA fallback, merged)
#   - https://github.com/karpathy/nanochat/pull/518 (DGX Spark mod reference)
#
# The DGX Spark has 1 GPU (GB10 Grace Blackwell Superchip) with 128 GB unified memory.
# No Flash Attention 3 (falls back to SDPA), no FP8, single GPU (nproc=1).
#
# Expected throughput: ~1,600-3,000 tok/sec, ~4.6% MFU
# Expected d12 training time: ~15-20 hours (optimized settings)
#
# Usage:
#   bash runs/speedrun_dgx_spark.sh
#   # or in a screen session (recommended, training takes days):
#   screen -L -Logfile runs/dgx_spark.log -S spark bash runs/speedrun_dgx_spark.sh

# --- Prerequisites ---
# 1. Install CUDA 13.0+ toolkit (Triton's ptxas needs sm_121a support for GB10):
#    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-ubuntu2404.pin
#    sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
#    wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb
#    sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_arm64.deb
#    sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
#    sudo apt-get update
#    sudo apt-get -y install cuda-toolkit-13-0
#
# 2. Set these env vars (add to ~/.bashrc for persistence):
#    export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
#    export CUDA_HOME=/usr/local/cuda-13.0
#    export PATH=/usr/local/cuda-13.0/bin:$PATH
#    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Ensure CUDA 13.0 env vars are set for Triton/ptxas
if [ -d "/usr/local/cuda-13.0" ]; then
    export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"; }
# Ensure uv is on PATH (needed if just installed or in a fresh shell)
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

python -m nanochat.dataset -n 8
# Download more shards in background (100 sufficient for d12 single-GPU training)
python -m nanochat.dataset -n 100 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# DGX Spark optimized: single GPU, no --fp8 (GB10 doesn't support it in this codebase).
#   --depth=12              GPT-1 scale, ~3.5x faster than d20
#   --device-batch-size=64  better GPU saturation with 128 GB unified mem
#   --max-seq-len=1024      halved context, quadratic attention savings
#   --window-pattern=L      avoids sliding window overhead without FA3
#   --core-metric-every=-1  skip expensive CORE eval mid-train
#   --sample-every=-1       skip sampling mid-train
#   --eval-every=500        less frequent val loss checks
#   --save-every=-1         only save final checkpoint
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --device-batch-size=64 \
    --max-seq-len=1024 \
    --window-pattern=L \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --eval-every=500 \
    --save-every=-1 \
    --run=$WANDB_RUN

# evaluate the model
python -m scripts.base_eval --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
python -m nanochat.report generate
