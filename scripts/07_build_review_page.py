"""Generate an interactive HTML page for reviewing the 284 flagged clips.

Output: data/review.html — self-contained (no network). Open by running:
    uv run python -m http.server -d data 8765
then visit http://localhost:8765/review.html

Features:
  - Embedded <audio> players with keyboard shortcuts (Space = play/pause)
  - Editable transcript per clip, pre-filled with canonical text
  - Include/Exclude toggle per clip (default by bucket)
  - Filter by bucket, progress counter
  - Decisions persist to localStorage, exportable as JSON via "Download decisions"
  - Apply with: uv run python scripts/08_apply_review.py --decisions path/to/decisions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BUCKET_PRIORITY = {
    "inferred_sfx": 0,
    "inline_annotation": 1,
    "only_in_cohere": 2,
    "annotation_only": 3,
    "major_diff": 4,
}

BUCKET_DEFAULT_INCLUDE = {
    "inferred_sfx": False,
    "inline_annotation": False,
    "annotation_only": False,
    "only_in_cohere": True,
    "major_diff": True,
}

BUCKET_COLORS = {
    "inferred_sfx": "#c0392b",
    "inline_annotation": "#d35400",
    "only_in_cohere": "#2980b9",
    "annotation_only": "#7f8c8d",
    "major_diff": "#27ae60",
}


def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;"))


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--recon", type=Path, default=root / "data" / "reconciliation.jsonl")
    ap.add_argument("--out", type=Path, default=root / "data" / "review.html")
    ap.add_argument("--data-root", type=Path, default=root / "data",
                    help="HTTP root the page will be served from (audio paths relativized to this)")
    args = ap.parse_args()

    recs = []
    for line in args.recon.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["bucket"] in BUCKET_PRIORITY:
            recs.append(r)

    recs.sort(key=lambda r: (BUCKET_PRIORITY[r["bucket"]], r["filename"]))

    rows_html = []
    for r in recs:
        audio_path = Path(r["audio"]).relative_to(args.data_root)
        default_include = BUCKET_DEFAULT_INCLUDE[r["bucket"]]
        wiki_html = html_escape(r["wiki_text"] or "")
        cohere_html = html_escape(r["cohere_text"] or "")
        canonical_html = html_escape(r["canonical_text"] or "")
        chapter = html_escape(r.get("wiki_chapter") or "")
        wer = f"{r['wer']:.2f}" if r.get("wer") is not None else "—"
        color = BUCKET_COLORS[r["bucket"]]

        rows_html.append(f"""
<tr data-filename="{html_escape(r['filename'])}" data-bucket="{r['bucket']}" data-default-include="{str(default_include).lower()}">
  <td class="num"></td>
  <td class="file">
    <code>{html_escape(r['filename'])}</code>
    <div class="meta">
      <span class="bucket" style="background:{color}">{r['bucket']}</span>
      <span class="wer">WER {wer}</span>
      <span class="chapter">{chapter}</span>
    </div>
  </td>
  <td class="audio"><audio controls preload="none" src="{html_escape(str(audio_path))}"></audio></td>
  <td class="texts">
    <div class="labelled"><label>wiki</label><div class="ro">{wiki_html or '<em>—</em>'}</div></div>
    <div class="labelled"><label>cohere</label><div class="ro">{cohere_html}</div></div>
    <div class="labelled"><label>canonical (edit)</label>
      <textarea class="edit" rows="2">{canonical_html}</textarea></div>
  </td>
  <td class="decision">
    <label><input type="radio" name="inc-{html_escape(r['filename'])}" value="true" {'checked' if default_include else ''}> keep</label><br>
    <label><input type="radio" name="inc-{html_escape(r['filename'])}" value="false" {'' if default_include else 'checked'}> drop</label>
  </td>
</tr>""")

    bucket_counts = {}
    for r in recs:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1

    buckets_summary = " | ".join(
        f'<span style="color:{BUCKET_COLORS[b]}"><b>{b}</b>: {n}</span>'
        for b, n in sorted(bucket_counts.items(), key=lambda x: BUCKET_PRIORITY[x[0]])
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>GLaDOS dataset review — {len(recs)} clips</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #fafafa; color: #222; }}
  h1 {{ margin: 0 0 4px; }}
  .top {{ position: sticky; top: 0; background: #fafafa; padding: 10px 0; border-bottom: 1px solid #ccc; z-index: 10; }}
  .summary {{ font-size: 13px; color: #555; margin: 6px 0; }}
  button {{ padding: 6px 14px; margin-right: 8px; font-size: 14px; cursor: pointer; }}
  select {{ padding: 4px; font-size: 14px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; background: white; }}
  td {{ padding: 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  td.num {{ width: 40px; color: #aaa; text-align: right; font-variant: tabular-nums; }}
  td.file {{ width: 280px; }}
  td.file code {{ font-size: 12px; word-break: break-all; }}
  td.file .meta {{ font-size: 11px; color: #666; margin-top: 4px; }}
  td.file .bucket {{ color: white; padding: 1px 6px; border-radius: 3px; margin-right: 6px; }}
  td.file .wer {{ margin-right: 6px; }}
  td.audio {{ width: 260px; }}
  td.audio audio {{ width: 100%; height: 32px; }}
  td.texts {{ font-size: 13px; }}
  td.texts .labelled {{ display: grid; grid-template-columns: 70px 1fr; gap: 8px; margin-bottom: 4px; align-items: start; }}
  td.texts label {{ color: #888; font-size: 11px; text-transform: uppercase; padding-top: 4px; }}
  td.texts .ro {{ background: #f4f4f4; padding: 4px 6px; border-radius: 3px; min-height: 18px; }}
  td.texts textarea {{ width: 100%; font-family: inherit; font-size: 13px; padding: 4px; border: 1px solid #aaa; border-radius: 3px; resize: vertical; }}
  td.decision {{ width: 80px; font-size: 13px; }}
  tr.hidden {{ display: none; }}
  tr.keep {{ background: rgba(39, 174, 96, 0.04); }}
  tr.drop {{ background: rgba(192, 57, 43, 0.05); }}
  tr.edited .edit {{ border-color: #e67e22; box-shadow: 0 0 0 2px rgba(230,126,34,0.15); }}
  .kbd-help {{ font-size: 11px; color: #999; margin-top: 4px; }}
</style>
</head><body>
<h1>GLaDOS dataset review</h1>
<div class="summary">{buckets_summary}</div>
<div class="top">
  <button id="download">Download decisions</button>
  <button id="clear">Reset all</button>
  Filter:
  <select id="filter">
    <option value="all">all ({len(recs)})</option>
"""
    for b, n in sorted(bucket_counts.items(), key=lambda x: BUCKET_PRIORITY[x[0]]):
        html += f'    <option value="{b}">{b} ({n})</option>\n'
    html += """  </select>
  <span id="progress" class="summary"></span>
  <div class="kbd-help">Shortcuts: <b>J</b>/<b>K</b> next/prev · <b>Space</b> play · <b>Y</b> keep · <b>N</b> drop · <b>E</b> focus edit</div>
</div>
<table>
<tbody id="rows">
"""
    html += "\n".join(rows_html)
    html += """
</tbody>
</table>
<script>
const KEY = 'glados-review-v1';
function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
function save(state) { localStorage.setItem(KEY, JSON.stringify(state)); }

const state = load();
const rows = document.querySelectorAll('#rows tr');

// Initialize from localStorage + defaults
rows.forEach((tr, idx) => {
  tr.querySelector('.num').textContent = (idx+1) + '.';
  const fn = tr.dataset.filename;
  const defaultInc = tr.dataset.defaultInclude === 'true';
  const s = state[fn] || {};
  const include = s.include !== undefined ? s.include : defaultInc;
  tr.querySelector(`input[value="${include ? 'true' : 'false'}"]`).checked = true;
  tr.classList.toggle('keep', include);
  tr.classList.toggle('drop', !include);
  const ta = tr.querySelector('textarea.edit');
  if (s.text !== undefined) ta.value = s.text;
  const origText = ta.defaultValue;
  if (ta.value !== origText) tr.classList.add('edited');

  // Wire decision radios
  tr.querySelectorAll('input[type=radio]').forEach(r => {
    r.addEventListener('change', () => {
      const keep = r.value === 'true';
      state[fn] = { ...(state[fn]||{}), include: keep };
      tr.classList.toggle('keep', keep);
      tr.classList.toggle('drop', !keep);
      save(state);
      updateProgress();
    });
  });
  ta.addEventListener('input', () => {
    state[fn] = { ...(state[fn]||{}), text: ta.value };
    tr.classList.toggle('edited', ta.value !== origText);
    save(state);
  });
});

// Filter
document.getElementById('filter').addEventListener('change', e => {
  const val = e.target.value;
  rows.forEach(tr => {
    tr.classList.toggle('hidden', val !== 'all' && tr.dataset.bucket !== val);
  });
  currentIdx = firstVisible();
  updateProgress();
});

function firstVisible() {
  for (let i = 0; i < rows.length; i++) if (!rows[i].classList.contains('hidden')) return i;
  return 0;
}
function visibleRows() { return Array.from(rows).filter(r => !r.classList.contains('hidden')); }
function updateProgress() {
  const total = visibleRows().length;
  const reviewed = visibleRows().filter(r => {
    const fn = r.dataset.filename;
    return state[fn] && state[fn].include !== undefined;
  }).length;
  document.getElementById('progress').textContent = `${reviewed} / ${total} reviewed`;
}
updateProgress();

// Keyboard navigation
let currentIdx = 0;
function focusRow(i) {
  const vis = visibleRows();
  if (!vis.length) return;
  i = Math.max(0, Math.min(vis.length - 1, i));
  currentIdx = i;
  vis[i].scrollIntoView({block: 'center', behavior: 'smooth'});
  vis[i].style.outline = '2px solid #3498db';
  vis.forEach((r, j) => { if (j !== i) r.style.outline = 'none'; });
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  const vis = visibleRows();
  const cur = vis[currentIdx];
  if (!cur) return;
  const fn = cur.dataset.filename;
  if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); focusRow(currentIdx + 1); }
  else if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); focusRow(currentIdx - 1); }
  else if (e.key === ' ') { e.preventDefault(); const a = cur.querySelector('audio'); a.paused ? a.play() : a.pause(); }
  else if (e.key === 'y' || e.key === 'Y') { cur.querySelector('input[value="true"]').click(); }
  else if (e.key === 'n' || e.key === 'N') { cur.querySelector('input[value="false"]').click(); }
  else if (e.key === 'e' || e.key === 'E') { e.preventDefault(); cur.querySelector('textarea.edit').focus(); }
});

document.getElementById('download').addEventListener('click', () => {
  // Include default decisions for every row so the apply script is fully deterministic
  const decisions = {};
  rows.forEach(tr => {
    const fn = tr.dataset.filename;
    const defaultInc = tr.dataset.defaultInclude === 'true';
    const s = state[fn] || {};
    const include = s.include !== undefined ? s.include : defaultInc;
    const origText = tr.querySelector('textarea.edit').defaultValue;
    const text = (s.text !== undefined && s.text !== origText) ? s.text : null;
    decisions[fn] = { include, text };
  });
  const blob = new Blob([JSON.stringify(decisions, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'review_decisions.json';
  a.click();
});

document.getElementById('clear').addEventListener('click', () => {
  if (confirm('Reset all decisions and edits?')) {
    localStorage.removeItem(KEY);
    location.reload();
  }
});

focusRow(0);
</script>
</body></html>
"""
    args.out.write_text(html)
    print(f"Wrote {args.out} ({len(recs)} rows)")
    print(f"Serve with: uv run python -m http.server -d {args.data_root} 8765")
    print(f"Open:       http://localhost:8765/review.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
