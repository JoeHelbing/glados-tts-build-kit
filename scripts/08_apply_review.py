"""Apply decisions from the HTML review page back to reconciliation.jsonl.

Input:  data/reconciliation.jsonl + review_decisions.json (from review.html export)
Output: data/reconciliation.jsonl (overwritten) with `include` and `canonical_text`
        updated per user decisions. Original file is backed up to .bak first.

After this, rerun 04_build_manifest.py to regenerate manifest.jsonl.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--decisions", type=Path, required=True, help="path to downloaded review_decisions.json")
    ap.add_argument("--recon", type=Path, default=root / "data" / "reconciliation.jsonl")
    args = ap.parse_args()

    decisions = json.loads(args.decisions.read_text())
    records = [json.loads(l) for l in args.recon.read_text().splitlines() if l.strip()]

    backup = args.recon.with_suffix(args.recon.suffix + ".bak")
    shutil.copy2(args.recon, backup)

    applied = {"include_change": 0, "text_edit": 0, "no_change": 0, "not_in_decisions": 0}
    with args.recon.open("w") as f:
        for r in records:
            d = decisions.get(r["filename"])
            if d is None:
                applied["not_in_decisions"] += 1
                f.write(json.dumps(r) + "\n")
                continue
            changed = False
            if d.get("include") is not None and d["include"] != r.get("include"):
                r["include"] = bool(d["include"])
                applied["include_change"] += 1
                changed = True
            if d.get("text"):
                r["canonical_text"] = d["text"]
                r["text_source"] = "human_edit"
                applied["text_edit"] += 1
                changed = True
            if not changed:
                applied["no_change"] += 1
            f.write(json.dumps(r) + "\n")

    print(f"Backup: {backup}")
    print(f"Applied:")
    for k, v in applied.items():
        print(f"  {k}: {v}")
    print(f"\nRegenerate manifest with: uv run python scripts/04_build_manifest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
