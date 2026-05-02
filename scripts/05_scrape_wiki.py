"""Scrape Portal Wiki GLaDOS voice-line pages into a filename -> canonical text map.

Two pages:
  - https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal)
  - https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal_2)

Each <li> contains a transcript in italicized quotes followed by a Download link
pointing at the raw wav. We match the audio URL's basename against our corpus.

Annotation-only transcripts like "[train horn]" or "[fast gibberish]" mean the
clip contains no usable speech — flagged and excluded from training.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


WIKI_URLS = {
    "portal1": "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal)",
    "portal2": "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal_2)",
    "coop": "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Cooperative_Testing_Initiative)",
    "other": "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Other)",
}

# "[train horn]", "[hums]", "[fast gibberish]" — wiki uses square brackets for SFX/context
# Italics in mediawiki render as <i>...</i>; quoted lines often _underscored_ but we work with plain text
ANNOTATION_PATTERN = re.compile(r"^\s*\[[^\]]+\]\s*$")
INLINE_ANNOTATION_PATTERN = re.compile(r"\[[^\]]+\]")


def normalize_filename(name: str) -> str:
    """Strip the GLaDOS_/glados_ wiki prefix and lowercase. Also drop any dir."""
    stem = Path(name).stem.lower()
    # wiki uses GLaDOS_<filename>.wav; game files omit that prefix
    for prefix in ("glados_", "announcer_", "cavejohnson_", "wheatley_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


class WikiParser(HTMLParser):
    """Extracts (transcript_text, audio_basename) pairs.

    Wiki pages nest <ul>/<li> — context notes (italicized) sometimes contain
    child <ul><li>...</li></ul> for the lines they describe. We push a frame
    per <li> so inner <li>s become their own records and outer <li>s don't
    swallow their children's audio.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[tuple[str, list[str], str, str]] = []
        self._li_stack: list[dict] = []  # stack of {"text": [...], "hrefs": [...]}
        self._in_heading: int | None = None
        self._heading_buf: list[str] = []
        self._chapter = ""
        self._section = ""

    def handle_starttag(self, tag: str, attrs: list) -> None:
        d = dict(attrs)
        if tag == "li":
            self._li_stack.append({"text": [], "hrefs": []})
        elif tag == "a" and self._li_stack:
            href = d.get("href", "")
            if href.endswith(".wav"):
                self._li_stack[-1]["hrefs"].append(href)
        elif tag in ("h2", "h3"):
            self._in_heading = 2 if tag == "h2" else 3
            self._heading_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._li_stack:
            frame = self._li_stack.pop()
            if frame["hrefs"]:
                text = " ".join(frame["text"]).strip()
                basenames = [Path(h).name for h in frame["hrefs"]]
                self.records.append((text, basenames, self._chapter, self._section))
        elif tag in ("h2", "h3") and self._in_heading:
            heading = "".join(self._heading_buf).strip()
            if self._in_heading == 2:
                self._chapter = heading
                self._section = ""
            else:
                self._section = heading
            self._in_heading = None
            self._heading_buf = []

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading_buf.append(data)
        elif self._li_stack:
            # accumulate only on innermost frame; it owns the nearest transcript
            self._li_stack[-1]["text"].append(data)


def clean_transcript(raw: str) -> str:
    """Strip wiki cruft around an italicized quoted transcript.

    Raw text from li looks like: ' "_Hello and, again, welcome to the ..._" | Download | Play '
    We strip: surrounding whitespace, the ` | ... | Download | Play` tail, wrapping underscores,
    wrapping quotes.
    """
    # Remove everything after the first '|' pipe — that's where the Download|Play columns start
    text = raw.split("|", 1)[0].strip()
    # Strip wrapping quotes (straight or curly)
    for q in ('"', "“", "”"):
        if text.startswith(q):
            text = text[len(q):]
        if text.endswith(q):
            text = text[:-len(q)]
    text = text.strip()
    # Strip wrapping underscores used as mediawiki italic markers in some renders
    text = text.strip("_").strip()
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", type=Path, default=root / "data" / "wiki.jsonl")
    ap.add_argument("--cache-dir", type=Path, default=root / "data" / ".wiki_cache")
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    all_records: dict[str, dict] = {}
    totals = {"pages": 0, "entries": 0, "with_audio": 0, "unique_files": 0, "annotation_only": 0}

    for key, url in WIKI_URLS.items():
        cache_file = args.cache_dir / f"{key}.html"
        if cache_file.exists():
            html = cache_file.read_text()
            print(f"[{key}] cached")
        else:
            print(f"[{key}] fetching {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "glados-tts-scraper/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode()
            cache_file.write_text(html)

        parser = WikiParser()
        parser.feed(html)
        totals["pages"] += 1

        for raw_text, basenames, chapter, section in parser.records:
            text = clean_transcript(raw_text)
            annotation_only = bool(ANNOTATION_PATTERN.match(text))
            has_inline_annotation = bool(INLINE_ANNOTATION_PATTERN.search(text)) and not annotation_only
            totals["entries"] += 1
            for bn in basenames:
                totals["with_audio"] += 1
                canon = normalize_filename(bn)
                # If same basename appears twice across pages, keep first non-empty
                existing = all_records.get(canon)
                if existing and existing["wiki_text"]:
                    continue
                all_records[canon] = {
                    "filename": canon,
                    "wiki_text": text,
                    "wiki_source": key,
                    "wiki_chapter": chapter,
                    "wiki_section": section,
                    "annotation_only": annotation_only,
                    "has_inline_annotation": has_inline_annotation,
                }
                if annotation_only:
                    totals["annotation_only"] += 1

    totals["unique_files"] = len(all_records)

    with args.out.open("w") as f:
        for rec in sorted(all_records.values(), key=lambda r: r["filename"]):
            f.write(json.dumps(rec) + "\n")

    print()
    print(f"Wrote {args.out}")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
