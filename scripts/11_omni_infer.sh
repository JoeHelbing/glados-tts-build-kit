#!/usr/bin/env bash
# Generate samples from an OmniVoice finetune checkpoint.
# Usage:
#   bash scripts/11_omni_infer.sh [CHECKPOINT_DIR] [OUT_DIR]
# Env overrides:
#   REF_AUDIO   reference clip to clone (default: a2_triple_laser01.wav)
#   REF_TEXT    transcript of REF_AUDIO
#   PROMPTS     path to AB prompts JSONL
#   STEPS       diffusion steps (default 32)
#   GUIDANCE    guidance scale (default 1.5)

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OMNI="${OMNI:-$HOME/git/OmniVoice}"
CHECKPOINT="${1:-$ROOT/data/omni/exp_finetune/checkpoint-5000}"
OUT_DIR="${2:-$ROOT/data/samples/omni_finetune}"
REF_AUDIO="${REF_AUDIO:-$ROOT/data/pcm/glados/a2_triple_laser01.wav}"
REF_TEXT="${REF_TEXT:-Federal regulations require me to warn you that this next test chamber is looking pretty good.}"
PROMPTS="${PROMPTS:-$ROOT/plan/ab_prompts.jsonl}"
STEPS="${STEPS:-32}"
GUIDANCE="${GUIDANCE:-1.5}"
PYTHON="${PYTHON:-$OMNI/.venv/bin/python}"

mkdir -p "$OUT_DIR"

ROOT="$(cd "$ROOT" && pwd)"
OMNI="$(cd "$OMNI" && pwd)"
CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
REF_AUDIO="$(cd "$(dirname "$REF_AUDIO")" && pwd)/$(basename "$REF_AUDIO")"
PROMPTS="$(cd "$(dirname "$PROMPTS")" && pwd)/$(basename "$PROMPTS")"
PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"

# Read each prompt and generate a wav
"$PYTHON" -c "
import json, subprocess, sys, time
prompts = [json.loads(l) for l in open('$PROMPTS') if l.strip()]
for p in prompts:
    out = '$OUT_DIR/' + p['id'] + '.wav'
    print(f'> {p[\"id\"]}: {p[\"text\"][:60]}...')
    t0 = time.time()
    subprocess.run([
        '$PYTHON', '-m', 'omnivoice.cli.infer',
        '--model', '$CHECKPOINT',
        '--text', p['text'],
        '--output', out,
        '--ref_audio', '$REF_AUDIO',
        '--ref_text', '$REF_TEXT',
        '--language', 'en',
        '--num_step', '$STEPS',
        '--guidance_scale', '$GUIDANCE',
    ], check=True, cwd='$OMNI')
    print(f'  done in {time.time()-t0:.1f}s -> {out}')
"
echo "All samples in $OUT_DIR"
