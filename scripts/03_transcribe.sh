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
COHEREX_DIR="${COHEREX_DIR:-$HOME/git/CohereX}"
BATCH="${BATCH:-32}"

[[ -d "$IN" ]] || {
  echo "ERROR: input dir $IN missing; run 02_transcode_pcm.sh first" >&2
  exit 1
}
[[ -d "$COHEREX_DIR" ]] || {
  echo "CohereX not cloned. Run:"
  echo "  git clone https://github.com/Diffio-AI/CohereX.git $COHEREX_DIR"
  echo "  (cd $COHEREX_DIR && uv sync)"
  exit 1
}
command -v uv >/dev/null || { echo "ERROR: uv not installed" >&2; exit 1; }

mkdir -p "$OUT"
IN="$(cd "$IN" && pwd)"
OUT="$(cd "$OUT" && pwd)"
COHEREX_DIR="$(cd "$COHEREX_DIR" && pwd)"
FILELIST=$(mktemp)
trap 'rm -f "$FILELIST"' EXIT
# Null-delimited to survive spaces/quotes in filenames (Portal 1 has at least one).
find "$IN" -name '*.wav' -print0 | sort -z > "$FILELIST"
N=$(tr -cd '\0' < "$FILELIST" | wc -c)
echo "Transcribing $N files to $OUT"
[[ "$N" -gt 0 ]] || {
  echo "ERROR: no WAV files found under $IN; run 02_transcode_pcm.sh first" >&2
  exit 1
}

cd "$COHEREX_DIR"
xargs -0 -a "$FILELIST" uv run coherex \
  --language en \
  --output_dir "$OUT" \
  --output_format json \
  --vad_method none \
  --batch_size "$BATCH" \
  --log-level warning

echo "Transcribed: $(find "$OUT" -name '*.json' | wc -l) / $N"
