"""Build the training manifest from the reconciliation output.

Consumes data/reconciliation.jsonl (produced by 06_reconcile.py) and emits:
  data/manifest.jsonl  — {audio, text, duration} records for Qwen3-TTS / F5-TTS

Only clips with `include=true` are kept. Also applies duration and min-char sanity.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--recon", type=Path, default=root / "data" / "reconciliation.jsonl")
    ap.add_argument("--out", type=Path, default=root / "data" / "manifest.jsonl")
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=15.0)
    ap.add_argument("--min-chars", type=int, default=3)
    args = ap.parse_args()

    kept = 0
    dropped = {"include_false": 0, "duration": 0, "min_chars": 0}
    durations = []

    with args.out.open("w") as f:
        for line in args.recon.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("include"):
                dropped["include_false"] += 1
                continue
            text = r["canonical_text"].strip()
            if len(text) < args.min_chars:
                dropped["min_chars"] += 1
                continue
            dur = ffprobe_duration(Path(r["audio"]))
            if dur < args.min_duration or dur > args.max_duration:
                dropped["duration"] += 1
                continue
            rec = {
                "audio": r["audio"],
                "text": text,
                "duration": round(dur, 3),
            }
            f.write(json.dumps(rec) + "\n")
            kept += 1
            durations.append(dur)

    total_min = sum(durations) / 60.0
    mean_dur = sum(durations) / len(durations) if durations else 0.0
    print(f"Wrote {args.out}")
    print(f"  kept:      {kept}")
    print(f"  dropped:   {dropped}")
    print(f"  total:     {total_min:.1f} min across {len(durations)} clips")
    print(f"  mean dur:  {mean_dur:.2f}s   min={min(durations):.2f}s   max={max(durations):.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
