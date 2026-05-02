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
GENERATED_DATA_CONFIG="${GENERATED_DATA_CONFIG:-$OUTPUT_DIR/omnivoice_data_config.generated.json}"
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
STAGE="${STAGE:-1}"
STOP_STAGE="${STOP_STAGE:-1}"
LOG="${LOG:-$OUTPUT_DIR/train.log}"
OMNI_PYTHON="${OMNI_PYTHON:-$OMNI/.venv/bin/python}"

[[ -d "$ROOT" ]] || { echo "ERROR: missing repo root: $ROOT" >&2; exit 1; }
ROOT="$(cd "$ROOT" && pwd)"
[[ -d "$OMNI" ]] || {
  echo "ERROR: OmniVoice not cloned at $OMNI" >&2
  echo "Clone/setup OmniVoice, then rerun with OMNI=/path/to/OmniVoice" >&2
  exit 1
}
OMNI="$(cd "$OMNI" && pwd)"

case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; esac
case "$TOKEN_DIR" in /*) ;; *) TOKEN_DIR="$ROOT/$TOKEN_DIR" ;; esac
case "$TRAIN_JSONL" in /*) ;; *) TRAIN_JSONL="$ROOT/$TRAIN_JSONL" ;; esac
case "$DEV_JSONL" in ""|/*) ;; *) DEV_JSONL="$ROOT/$DEV_JSONL" ;; esac
case "$TRAIN_CONFIG" in /*) ;; *) TRAIN_CONFIG="$ROOT/$TRAIN_CONFIG" ;; esac
case "$DATA_CONFIG" in /*) ;; *) DATA_CONFIG="$ROOT/$DATA_CONFIG" ;; esac
case "$GENERATED_DATA_CONFIG" in /*) ;; *) GENERATED_DATA_CONFIG="$ROOT/$GENERATED_DATA_CONFIG" ;; esac
OMNI_PYTHON="${OMNI_PYTHON/#\~/$HOME}"
case "$OMNI_PYTHON" in /*) ;; *) OMNI_PYTHON="$OMNI/$OMNI_PYTHON" ;; esac

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
mkdir -p "$TOKEN_DIR"
TOKEN_DIR="$(cd "$TOKEN_DIR" && pwd)"
GENERATED_DATA_CONFIG="$(dirname "$GENERATED_DATA_CONFIG")/$(basename "$GENERATED_DATA_CONFIG")"

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
  [[ -f "$TRAIN_CONFIG" ]] || { echo "ERROR: missing TRAIN_CONFIG: $TRAIN_CONFIG" >&2; exit 1; }
  [[ -f "$DATA_CONFIG" ]] || { echo "ERROR: missing DATA_CONFIG: $DATA_CONFIG" >&2; exit 1; }
  "$OMNI_PYTHON" - "$DATA_CONFIG" "$GENERATED_DATA_CONFIG" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).expanduser().resolve()
dest = Path(sys.argv[2]).expanduser()
root = Path(sys.argv[3]).expanduser().resolve()
config = json.loads(source.read_text(encoding="utf-8"))

for split in ("train", "dev"):
    for item in config.get(split, []):
        paths = item.get("manifest_path")
        if paths is None:
            continue
        if isinstance(paths, str):
            paths = [paths]
        item["manifest_path"] = [
            str((root / path).resolve() if not Path(path).expanduser().is_absolute() else Path(path).expanduser().resolve())
            for path in paths
        ]

dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
  echo "Launching OmniVoice finetune"
  echo "  output:       $OUTPUT_DIR"
  echo "  train config: $TRAIN_CONFIG"
  echo "  data config:  $GENERATED_DATA_CONFIG"
  echo "  log:          $LOG"
  cd "$OMNI"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    accelerate launch \
      --gpu_ids "$GPU_IDS" \
      --num_processes "$NUM_GPUS" \
      -m omnivoice.cli.train \
      --train_config "$TRAIN_CONFIG" \
      --data_config "$GENERATED_DATA_CONFIG" \
      --output_dir "$OUTPUT_DIR" \
      2>&1 | tee "$LOG"
fi
