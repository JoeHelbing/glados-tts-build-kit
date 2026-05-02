"""Reconcile Cohere Transcribe output against Portal Wiki canonical transcripts.

For each clip:
  - wiki + cohere   → compute WER, pick canonical text (wiki wins)
  - wiki only       → canonical = wiki (shouldn't happen — wiki must have audio in game)
  - cohere only     → canonical = cohere, flagged only_in_cohere for manual review
  - annotation-only wiki entry ([train horn], [hums...]) → mark excluded

Outputs:
  data/reconciliation.jsonl  — one record per clip (all 1400), with canonical text + flags
  data/diff_review.csv       — only cases needing eyeballs, sorted by WER desc
  data/quality_report.md     — summary counts for the blog post
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def normalize_for_wer(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace — then tokenize for WER."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def word_error_rate(ref: str, hyp: str) -> float:
    """Standard Levenshtein-based WER at the word level. Returns [0, >1] for bad hyp."""
    r = normalize_for_wer(ref)
    h = normalize_for_wer(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein DP
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / m


def cohere_text(transcript_json: dict) -> str:
    return " ".join(s["text"].strip() for s in transcript_json.get("segments", [])).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--pcm-dir", type=Path, default=root / "data" / "pcm")
    ap.add_argument("--transcripts-dir", type=Path, default=root / "data" / "transcripts")
    ap.add_argument("--wiki", type=Path, default=root / "data" / "wiki.jsonl")
    ap.add_argument("--out", type=Path, default=root / "data" / "reconciliation.jsonl")
    ap.add_argument("--diff-csv", type=Path, default=root / "data" / "diff_review.csv")
    ap.add_argument("--report", type=Path, default=root / "data" / "quality_report.md")
    ap.add_argument("--minor-wer-max", type=float, default=0.10, help="WER below this is 'minor' — auto-accept wiki silently")
    args = ap.parse_args()

    wiki = {}
    for line in args.wiki.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        wiki[r["filename"]] = r

    stems = {p.stem.lower(): p for p in args.pcm_dir.rglob("*.wav")}

    # Filename patterns that indicate non-speech content (battle grunts, SFX, laughter).
    SFX_FILENAME_PATTERNS = (
        "anger", "scream", "pain", "laugh", "cough", "chuckle", "hit_nag",
        "ding_on", "ding_off", "death", "spheredestroy", "spheredrop",
        "ballhitpain", "_anger", "_rage",
    )

    def looks_like_sfx(stem: str, cohere: str) -> bool:
        low = stem.lower()
        if any(p in low for p in SFX_FILENAME_PATTERNS):
            return True
        # Single-char or one-word hallucinations (Cohere stabs at silence/noise)
        words = cohere.strip().split()
        if len(words) <= 1 and len(cohere.strip()) <= 5:
            return True
        return False

    recs = []
    for j in sorted(args.transcripts_dir.glob("*.json")):
        stem = j.stem.lower()
        wav = stems.get(stem)
        if wav is None:
            continue  # transcript without matching audio (shouldn't happen)
        transcript = json.loads(j.read_text())
        cohere = cohere_text(transcript)

        w = wiki.get(stem)
        rec = {
            "filename": stem,
            "audio": str(wav),
            "cohere_text": cohere,
            "wiki_text": w["wiki_text"] if w else None,
            "wiki_chapter": w["wiki_chapter"] if w else None,
            "annotation_only": bool(w and w["annotation_only"]),
            "has_inline_annotation": bool(w and w["has_inline_annotation"]),
        }
        if w:
            rec["wer"] = word_error_rate(w["wiki_text"], cohere)
            if w["annotation_only"]:
                rec["canonical_text"] = w["wiki_text"]
                rec["text_source"] = "wiki-annotation"
                rec["include"] = False
                rec["bucket"] = "annotation_only"
            elif w["has_inline_annotation"]:
                # Wiki marked [bzzt]/[cough]/[garbled] inline — audio has non-speech content
                # that would confuse training. Exclude by default; user can override.
                rec["canonical_text"] = w["wiki_text"]
                rec["text_source"] = "wiki"
                rec["include"] = False
                rec["bucket"] = "inline_annotation"
            elif rec["wer"] == 0.0:
                rec["canonical_text"] = w["wiki_text"]
                rec["text_source"] = "wiki"
                rec["include"] = True
                rec["bucket"] = "exact_match"
            elif rec["wer"] <= args.minor_wer_max:
                rec["canonical_text"] = w["wiki_text"]
                rec["text_source"] = "wiki"
                rec["include"] = True
                rec["bucket"] = "minor_diff"
            else:
                rec["canonical_text"] = w["wiki_text"]
                rec["text_source"] = "wiki"
                rec["include"] = True
                rec["bucket"] = "major_diff"
        else:
            rec["wer"] = None
            rec["canonical_text"] = cohere
            rec["text_source"] = "cohere"
            if looks_like_sfx(stem, cohere):
                rec["include"] = False
                rec["bucket"] = "inferred_sfx"
            else:
                rec["include"] = True
                rec["bucket"] = "only_in_cohere"

        recs.append(rec)

    # Bucket counts
    from collections import Counter
    buckets = Counter(r["bucket"] for r in recs)

    # Write reconciliation.jsonl
    with args.out.open("w") as f:
        for r in sorted(recs, key=lambda x: x["filename"]):
            f.write(json.dumps(r) + "\n")

    # Write diff_review.csv — only cases needing eyeballs
    review_buckets = {"major_diff", "only_in_cohere", "annotation_only", "inline_annotation", "inferred_sfx"}
    review = [r for r in recs if r["bucket"] in review_buckets]
    review.sort(key=lambda r: (r["bucket"], -(r.get("wer") or 0)))
    with args.diff_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "filename", "wer", "wiki_chapter", "wiki_text", "cohere_text", "audio"])
        for r in review:
            w.writerow([
                r["bucket"],
                r["filename"],
                f"{r['wer']:.3f}" if r["wer"] is not None else "",
                r.get("wiki_chapter") or "",
                r.get("wiki_text") or "",
                r["cohere_text"],
                r["audio"],
            ])

    # Quality report
    total = len(recs)
    wers = [r["wer"] for r in recs if r["wer"] is not None]
    mean_wer = sum(wers) / len(wers) if wers else 0.0
    report = f"""# Dataset quality report

Generated from {total} clips × {{cohere,wiki}}.

## Buckets

| Bucket | Count | % |
|---|---:|---:|
"""
    for b, n in sorted(buckets.items(), key=lambda x: -x[1]):
        report += f"| {b} | {n} | {100*n/total:.1f} |\n"
    report += f"""

## Cohere vs wiki

- Compared clips: **{len(wers)}** / {total}
- Mean WER (cohere vs wiki): **{mean_wer:.4f}** ({mean_wer*100:.2f}%)
- Exact matches: {buckets.get('exact_match', 0)}
- Minor diffs (WER ≤ {args.minor_wer_max*100:.0f}%): {buckets.get('minor_diff', 0)}
- Major diffs (WER > {args.minor_wer_max*100:.0f}%): {buckets.get('major_diff', 0)} — see diff_review.csv

## For manual review

{sum(1 for r in recs if r['bucket'] in review_buckets)} clips in `data/diff_review.csv`:
- `major_diff`: wiki and cohere disagree substantially
- `only_in_cohere`: no wiki entry (probably SFX/singing/cut line)
- `annotation_only`: wiki says `[train horn]` / `[hums...]` — auto-excluded from training

## Excluded from training

{sum(1 for r in recs if not r['include'])} clips flagged `include=false`.
"""
    args.report.write_text(report)
    print(f"Wrote {args.out}\nWrote {args.diff_csv} ({len(review)} rows)\nWrote {args.report}")
    print(f"\nBuckets: {dict(buckets)}")
    print(f"Mean Cohere-vs-wiki WER: {mean_wer*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
