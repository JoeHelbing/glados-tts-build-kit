#!/usr/bin/env bash
# Transcribe all GLaDOS clips with CohereX (Cohere Transcribe + wav2vec2 forced alignment).
#
# CohereX = WhisperX-shaped wrapper around CohereLabs/cohere-transcribe-03-2026.
# Produces one JSON per clip with segments, word-level timestamps, alignment scores.
#
# Expected runtime on RTX 3090: ~90 min for 1400 short clips.
set -euo pipefail

IN="${IN:-$(dirname "$0")/../data/pcm}"
OUT="${OUT:-$(dirname "$0")/../data/transcripts}"
COHEREX_DIR="${COHEREX_DIR:-$HOME/Downloads/yt/CohereX}"
BATCH="${BATCH:-32}"

[[ -d "$COHEREX_DIR" ]] || {
  echo "CohereX not cloned. Run:"
  echo "  git clone https://github.com/Diffio-AI/CohereX.git $COHEREX_DIR"
  echo "  (cd $COHEREX_DIR && uv sync)"
  exit 1
}

mkdir -p "$OUT"
FILELIST=$(mktemp)
# Null-delimited to survive spaces/quotes in filenames (Portal 1 has at least one).
find "$IN" -name '*.wav' -print0 | sort -z > "$FILELIST"
N=$(tr -cd '\0' < "$FILELIST" | wc -c)
echo "Transcribing $N files to $OUT"

cd "$COHEREX_DIR"
xargs -0 -a "$FILELIST" uv run coherex \
  --language en \
  --output_dir "$OUT" \
  --output_format json \
  --vad_method none \
  --batch_size "$BATCH" \
  --log-level warning

rm -f "$FILELIST"
echo "Transcribed: $(find "$OUT" -name '*.json' | wc -l) / $N"
