#!/usr/bin/env bash
# Run the full source-only GLaDOS TTS build pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_ENV="${PIPELINE_ENV:-$ROOT/config/pipeline.env}"
PIPELINE_ENV_EXPLICIT=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: bash scripts/train_glados_tts.sh [--env PATH] [--dry-run]

Runs extraction, PCM transcoding, CohereX transcription, Portal Wiki scraping,
reconciliation, manifest conversion, and OmniVoice tokenization/training.

Configure local paths by copying config/pipeline.env.example to
config/pipeline.env, or by exporting variables directly.

Important variables:
  STEAM_COMMON   Steam steamapps/common directory
  COHEREX_DIR    CohereX checkout with uv sync already run
  OMNI           OmniVoice checkout with its .venv already set up
  REVIEW_MODE    auto (default) or manual
  GPU_IDS        CUDA device ids passed to transcription/training
  STAGE          OmniVoice start stage: 0 tokenization, 1 training
  STOP_STAGE     OmniVoice stop stage: 0 tokenization, 1 training
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      PIPELINE_ENV="$2"
      PIPELINE_ENV_EXPLICIT=1
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -f "$PIPELINE_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$PIPELINE_ENV"
elif [[ "${PIPELINE_ENV_EXPLICIT:-0}" == "1" ]]; then
  echo "ERROR: env file not found: $PIPELINE_ENV" >&2
  exit 1
fi

STEAM_COMMON="${STEAM_COMMON:-$HOME/.local/share/Steam/steamapps/common}"
COHEREX_DIR="${COHEREX_DIR:-$HOME/git/CohereX}"
OMNI="${OMNI:-$HOME/git/OmniVoice}"
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
PAR="${PAR:-8}"
BATCH="${BATCH:-32}"
REVIEW_MODE="${REVIEW_MODE:-auto}"
STAGE="${STAGE:-0}"
STOP_STAGE="${STOP_STAGE:-1}"

need_command() {
  command -v "$1" >/dev/null || {
    echo "ERROR: missing required command: $1" >&2
    echo "Install prerequisites listed in README.md, then rerun." >&2
    exit 1
  }
}

need_path() {
  local label="$1"
  local path="$2"
  [[ -e "$path" ]] || {
    echo "ERROR: missing $label: $path" >&2
    exit 1
  }
}

run_step() {
  local label="$1"
  shift
  echo ""
  echo "==> $label"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '    '
    printf '%q ' "$@"
    echo
  else
    "$@"
  fi
}

case "$REVIEW_MODE" in
  auto|manual) ;;
  *)
    echo "ERROR: REVIEW_MODE must be 'auto' or 'manual' (got '$REVIEW_MODE')" >&2
    exit 1
    ;;
esac

need_command uv
need_command ffmpeg
need_command ffprobe
need_command vpk
need_path "Portal 2 VPK" "$STEAM_COMMON/Portal 2/portal2/pak01_dir.vpk"
need_path "Portal VPK" "$STEAM_COMMON/Portal/portal/portal_pak_dir.vpk"
need_path "CohereX checkout" "$COHEREX_DIR"
need_path "OmniVoice checkout" "$OMNI"
need_path "OmniVoice Python" "$OMNI/.venv/bin/python"

echo "GLaDOS TTS build pipeline"
echo "  repo:         $ROOT"
echo "  steam common: $STEAM_COMMON"
echo "  CohereX:      $COHEREX_DIR"
echo "  OmniVoice:    $OMNI"
echo "  review mode:  $REVIEW_MODE"
echo "  GPU ids:      $GPU_IDS"

run_step "Extract Portal voice assets" \
  env STEAM_COMMON="$STEAM_COMMON" bash "$ROOT/scripts/01_extract_vpks.sh"

run_step "Transcode to 24kHz mono PCM" \
  env PAR="$PAR" bash "$ROOT/scripts/02_transcode_pcm.sh"

run_step "Transcribe with CohereX" \
  env COHEREX_DIR="$COHEREX_DIR" BATCH="$BATCH" bash "$ROOT/scripts/03_transcribe.sh"

run_step "Scrape Portal Wiki transcripts" \
  uv run python "$ROOT/scripts/05_scrape_wiki.py"

run_step "Reconcile transcripts" \
  uv run python "$ROOT/scripts/06_reconcile.py"

if [[ "$REVIEW_MODE" == "manual" ]]; then
  run_step "Build manual review page" \
    uv run python "$ROOT/scripts/07_build_review_page.py"
  echo ""
  echo "Manual review requested. Start the review server with:"
  echo "  uv run python -m http.server -d data 8765"
  echo "Then open http://127.0.0.1:8765/review.html, export decisions, apply them, and rerun:"
  echo "  uv run python scripts/08_apply_review.py --decisions /path/to/review_decisions.json"
  echo "  uv run python scripts/04_build_manifest.py"
  echo "  uv run python scripts/09_convert_manifest.py --overwrite"
  echo "  OMNI=\"$OMNI\" STAGE=$STAGE STOP_STAGE=$STOP_STAGE bash scripts/10_train_omnivoice.sh"
  exit 0
fi

run_step "Build training manifest" \
  uv run python "$ROOT/scripts/04_build_manifest.py"

run_step "Convert manifest for OmniVoice" \
  uv run python "$ROOT/scripts/09_convert_manifest.py" --overwrite

run_step "Tokenize and train OmniVoice" \
  env OMNI="$OMNI" GPU_IDS="$GPU_IDS" NUM_GPUS="$NUM_GPUS" STAGE="$STAGE" STOP_STAGE="$STOP_STAGE" \
    bash "$ROOT/scripts/10_train_omnivoice.sh"

echo ""
echo "Training command completed. Checkpoints and logs are under data/omni/."
