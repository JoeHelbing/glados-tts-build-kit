#!/usr/bin/env bash
# Transcode Source Engine MP3-in-WAV files to real 24kHz mono PCM WAV.
#
# Source Engine .wav files are actually MP3-encoded (ffprobe shows codec_name=mp3).
# TTS training pipelines expect real PCM. 24kHz matches Qwen3-TTS / F5-TTS input.
set -euo pipefail

IN="${IN:-$(dirname "$0")/../data/raw/sound/vo}"
OUT="${OUT:-$(dirname "$0")/../data/pcm}"
PAR="${PAR:-8}"

command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not installed" >&2; exit 1; }
[[ -d "$IN" ]] || { echo "ERROR: input dir $IN missing; run 01_extract_vpks.sh first" >&2; exit 1; }

cd "$IN"
echo "Transcoding from $IN -> $OUT (24kHz mono PCM, parallelism=$PAR)"

find glados aperture_ai escape -name '*.wav' -print0 2>/dev/null \
  | xargs -0 -P "$PAR" -I{} bash -c '
    f="{}"
    out="'"$OUT"'/$f"
    mkdir -p "$(dirname "$out")"
    ffmpeg -v error -y -i "$f" -ar 24000 -ac 1 -c:a pcm_s16le "$out"
  '

echo ""
echo "Output:"
find "$OUT" -name '*.wav' | wc -l
du -sh "$OUT"
