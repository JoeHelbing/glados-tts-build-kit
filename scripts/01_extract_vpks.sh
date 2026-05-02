#!/usr/bin/env bash
# Extract GLaDOS voice lines from Portal 1 and Portal 2 VPK archives.
#
# Source Engine ships VO under sound/vo/<speaker>/*.wav inside the *_dir.vpk.
# Requires the `vpk` CLI (pip install vpk, or uv tool install vpk).
set -euo pipefail

STEAM_COMMON="${STEAM_COMMON:-$HOME/.local/share/Steam/steamapps/common}"
OUT="${OUT:-$(dirname "$0")/../data/raw}"

P2="$STEAM_COMMON/Portal 2/portal2/pak01_dir.vpk"
P1="$STEAM_COMMON/Portal/portal/portal_pak_dir.vpk"

for p in "$P1" "$P2"; do
  [[ -f "$p" ]] || { echo "ERROR: missing $p" >&2; exit 1; }
done

command -v vpk >/dev/null || { echo "ERROR: vpk CLI not installed. Run: uv tool install vpk" >&2; exit 1; }

mkdir -p "$OUT"
cd "$OUT"

echo "[1/3] Portal 2 glados/ ..."
vpk -x . -f 'sound/vo/glados/*' "$P2" >/dev/null

echo "[2/3] Portal 1 aperture_ai/ ..."
vpk -x . -f 'sound/vo/aperture_ai/*' "$P1" >/dev/null

echo "[3/3] Portal 1 escape/ ..."
vpk -x . -f 'sound/vo/escape/*' "$P1" >/dev/null

echo ""
echo "Extracted to $OUT/sound/vo/"
find "$OUT/sound/vo" -name '*.wav' | awk -F/ '{print $(NF-1)}' | sort | uniq -c
echo ""
echo "Total files: $(find "$OUT/sound/vo" -name '*.wav' | wc -l)"
