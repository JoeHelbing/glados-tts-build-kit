#!/usr/bin/env bash
# Launch an OmniVoice finetune. Defaults assume the upstream OmniVoice repo is
# cloned next to this repo and the manifest/config files live under ./data.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OMNI="${OMNI:-$HOME/git/OmniVoice}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/omni/exp_finetune}"
TOKEN_DIR="${TOKEN_DIR:-$ROOT/data/omni/tokens}"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT/data/omni_train.jsonl}"
DEV_JSONL="${DEV_JSONL:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-eustlb/higgs-audio-v2-tokenizer}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/config/omnivoice_train_config.json}"
DATA_CONFIG="${DATA_CONFIG:-$ROOT/config/omnivoice_data_config.json}"
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
STAGE="${STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-1}"
LOG="${LOG:-$OUTPUT_DIR/train.log}"
OMNI_PYTHON="${OMNI_PYTHON:-$OMNI/.venv/bin/python}"

mkdir -p "$OUTPUT_DIR"

cd "$OMNI"

export PYTHONPATH="$OMNI:${PYTHONPATH:-}"
export PATH="$OMNI/.venv/bin:$PATH"
[[ -x "$OMNI_PYTHON" ]] || { echo "ERROR: missing OmniVoice Python: $OMNI_PYTHON" >&2; exit 1; }

if [[ "$STAGE" -le 0 && "$STOP_STAGE" -ge 0 ]]; then
  echo "Stage 0: tokenizing audio"
  [[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: missing TRAIN_JSONL: $TRAIN_JSONL" >&2; exit 1; }
  for split_jsonl_path in "$TRAIN_JSONL" "$DEV_JSONL"; do
    [[ -n "$split_jsonl_path" ]] || continue
    if [[ "$split_jsonl_path" == "$TRAIN_JSONL" ]]; then
      split="train"
    else
      split="dev"
    fi
    echo "  Tokenizing $split from $split_jsonl_path"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      "$OMNI_PYTHON" -m omnivoice.scripts.extract_audio_tokens \
        --input_jsonl "$split_jsonl_path" \
        --tar_output_pattern "$TOKEN_DIR/$split/audios/shard-%06d.tar" \
        --jsonl_output_pattern "$TOKEN_DIR/$split/txts/shard-%06d.jsonl" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --nj_per_gpu 3 \
        --shuffle True
    echo "  Wrote $TOKEN_DIR/$split/data.lst"
  done
fi

if [[ "$STAGE" -le 1 && "$STOP_STAGE" -ge 1 ]]; then
  echo "Launching OmniVoice finetune"
  echo "  output:       $OUTPUT_DIR"
  echo "  train config: $TRAIN_CONFIG"
  echo "  data config:  $DATA_CONFIG"
  echo "  log:          $LOG"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    accelerate launch \
      --gpu_ids "$GPU_IDS" \
      --num_processes "$NUM_GPUS" \
      -m omnivoice.cli.train \
      --train_config "$TRAIN_CONFIG" \
      --data_config "$DATA_CONFIG" \
      --output_dir "$OUTPUT_DIR" \
      2>&1 | tee "$LOG"
fi
