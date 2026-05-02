#!/usr/bin/env python
"""Convert the local manifest into OmniVoice fine-tuning JSONL format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = DEFAULT_ROOT / "data" / "reconciliation.jsonl"
DEFAULT_OUTPUT = DEFAULT_ROOT / "data" / "omni_train.jsonl"


def make_absolute(path_text: str, base_dir: Path) -> Path:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def iter_manifest_rows(path: Path):
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            yield line_number, json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source manifest.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination OmniVoice raw JSONL")
    parser.add_argument("--language-id", default="en", help="Language id stored on each OmniVoice row")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input manifest not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path} (pass --overwrite to replace it)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    first_audio: str | None = None
    with output_path.open("w", encoding="utf-8") as handle:
        for line_number, row in iter_manifest_rows(input_path):
            if row.get("include") is False:
                skipped += 1
                continue

            text_value = row.get("text") or row.get("canonical_text") or row.get("wiki_text")
            missing = [key for key in ("audio",) if key not in row]
            if missing or not text_value:
                missing_text = ", ".join(missing + (["text/canonical_text/wiki_text"] if not text_value else []))
                raise SystemExit(f"{input_path}:{line_number}: missing required fields: {missing_text}")

            audio_path = make_absolute(str(row["audio"]), input_path.parent)
            if not audio_path.is_file():
                raise SystemExit(f"{input_path}:{line_number}: audio file not found: {audio_path}")

            converted = {
                "id": audio_path.stem,
                "audio_path": str(audio_path),
                "text": str(text_value),
                "language_id": args.language_id,
            }
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
            if first_audio is None:
                first_audio = converted["audio_path"]

    print(f"Wrote {output_path}")
    print(f"  rows: {count}")
    print(f"  skipped: {skipped}")
    if first_audio:
        print(f"  first_audio: {first_audio}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
