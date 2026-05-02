#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "data" / "samples"
COMPARE_DIR = SAMPLE_ROOT / "omni_compare"
GENERATED_DIR = COMPARE_DIR / "generated"
HISTORY_PATH = GENERATED_DIR / "history.jsonl"
EVAL_PATH = COMPARE_DIR / "listening_eval.jsonl"
OMNIVOICE_ROOT = Path(os.environ.get("OMNIVOICE_ROOT", Path.home() / "git" / "OmniVoice"))
OMNIVOICE_PYTHON = OMNIVOICE_ROOT / ".venv" / "bin" / "python"
CHECKPOINT_ROOT = Path(
    os.environ.get("OMNI_COMPARE_CHECKPOINT_ROOT", ROOT / "data" / "omni" / "exp_finetune")
)
REF_AUDIO = ROOT / "data" / "pcm" / "glados" / "a2_triple_laser01.wav"
REF_TEXT = (
    "Federal regulations require me to warn you that this next test chamber is "
    "looking pretty good."
)
HOST = "127.0.0.1"
PORT = 8771
MAX_TEXT_LEN = 700
MAX_SEEDS = 12
MAX_GENERATION_JOBS = 60
DEFAULT_SEEDS = [42]
DEFAULT_SPEED = 1.0
FALLBACK_CHECKPOINTS = ["checkpoint-1500", "checkpoint-2000", "checkpoint-2500"]
PINNED_COMPARISON_STEPS = [1500, 5000]
PROMPTS = [
    {
        "id": "prompt-01",
        "title": 'Oh. It is you...',
        "text": 'Oh. It is you. It has been a long time. How have you been?',
        "durations": {
            "checkpoint-1500": "4.12s",
            "checkpoint-2000": "4.12s",
            "checkpoint-2500": "4.12s",
        },
    },
    {
        "id": "prompt-02",
        "title": "plate tectonics...",
        "text": "Plate tectonics was not originally part of this test, but the floor has made several compelling arguments.",
        "durations": {
            "checkpoint-1500": "8.92s",
            "checkpoint-2000": "8.92s",
            "checkpoint-2500": "8.92s",
        },
    },
    {
        "id": "prompt-03",
        "title": "Please do not touch...",
        "text": "Please do not touch anything marked with a red warning label, a yellow warning label, or a label that has been removed for your convenience.",
        "durations": {
            "checkpoint-1500": "8.85s",
            "checkpoint-2000": "9.28s",
            "checkpoint-2500": "9.08s",
        },
    },
    {
        "id": "prompt-04",
        "title": "quick brown fox...",
        "text": "The quick brown fox jumps over the lazy dog, then files a complaint about the test conditions.",
        "durations": {
            "checkpoint-1500": "8.04s",
            "checkpoint-2000": "8.04s",
            "checkpoint-2500": "8.04s",
        },
    },
    {
        "id": "prompt-05",
        "title": "less disappointing...",
        "text": "That was less disappointing than usual. Please do not interpret this as praise.",
        "durations": {
            "checkpoint-1500": "9.69s",
            "checkpoint-2000": "9.80s",
            "checkpoint-2500": "9.80s",
        },
    },
]


def checkpoint_step(name: str) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if not match:
        return 0
    return int(match.group(1))


def parse_eval_losses(run_root: Path) -> dict[int, float]:
    log_path = run_root / "train.log"
    if not log_path.exists():
        return {}
    losses: dict[int, float] = {}
    pending_step: int | None = None
    step_re = re.compile(r"Running evaluation at step (\d+)")
    loss_re = re.compile(r"Eval Loss:\s*([0-9.]+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = step_re.search(line)
        if step_match:
            pending_step = int(step_match.group(1))
            continue
        loss_match = loss_re.search(line)
        if loss_match and pending_step is not None:
            losses[pending_step] = float(loss_match.group(1))
            pending_step = None
    return losses


def select_checkpoints(run_root: Path = CHECKPOINT_ROOT) -> list[dict[str, Any]]:
    dirs = sorted(
        [
            path.name
            for path in run_root.glob("checkpoint-*")
            if path.is_dir() and checkpoint_step(path.name) > 0
        ],
        key=checkpoint_step,
    )
    if not dirs:
        dirs = FALLBACK_CHECKPOINTS

    losses = parse_eval_losses(run_root)
    available_steps = [checkpoint_step(name) for name in dirs]
    loss_steps = [step for step in available_steps if step in losses]

    if loss_steps:
        best_step = min(loss_steps, key=lambda step: losses[step])
        lower = [step for step in available_steps if step < best_step]
        higher = [step for step in available_steps if step > best_step]
        selected_steps = []
        if lower:
            selected_steps.append(max(lower))
        selected_steps.append(best_step)
        if higher:
            selected_steps.append(min(higher))

        if len(selected_steps) < 3:
            extras = sorted(
                [step for step in available_steps if step not in selected_steps],
                key=lambda step: (
                    losses.get(step, float("inf")),
                    abs(step - best_step),
                    step,
                ),
            )
            selected_steps.extend(extras[: 3 - len(selected_steps)])
    else:
        selected_steps = available_steps[:3]

    selected_steps = sorted(
        set(selected_steps)
        | {step for step in PINNED_COMPARISON_STEPS if step in available_steps}
    )
    if len(selected_steps) < 3:
        for step in available_steps:
            if step not in selected_steps:
                selected_steps.append(step)
            if len(selected_steps) == 3:
                break
        selected_steps.sort()

    best_eval_step = min(loss_steps, key=lambda step: losses[step]) if loss_steps else None
    return [
        {
            "name": f"checkpoint-{step}",
            "step": step,
            "eval_loss": losses.get(step),
            "is_best_eval": step == best_eval_step,
        }
        for step in selected_steps
    ]


CHECKPOINT_META = select_checkpoints()
CHECKPOINTS = [checkpoint["name"] for checkpoint in CHECKPOINT_META]


@dataclass(frozen=True)
class GeneratedSample:
    checkpoint: str
    seed: int
    speed: float
    text: str
    created_at: str
    elapsed_seconds: float
    path: str
    url: str


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_job: dict[str, Any] | None = None


STATE = ServerState()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str, *, limit: int = 48) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    parts = [part for part in cleaned.split("-") if part]
    slug = "-".join(parts)
    return (slug[:limit].strip("-") or "sample")


def sample_url(path: Path) -> str:
    return f"/samples/{path.relative_to(SAMPLE_ROOT).as_posix()}"


def wav_duration(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            seconds = handle.getnframes() / float(handle.getframerate())
    except (wave.Error, OSError, ZeroDivisionError):
        return None
    return f"{seconds:.2f}s"


def comparison_matrix() -> list[dict[str, Any]]:
    latest_eval = latest_eval_by_sample(read_eval_records())
    rows: list[dict[str, Any]] = []
    for prompt in PROMPTS:
        row = {
            "id": prompt["id"],
            "title": prompt["title"],
            "text": prompt["text"],
            "samples": {},
        }
        for checkpoint in CHECKPOINTS:
            path = COMPARE_DIR / f"{checkpoint}_{prompt['id']}.wav"
            row["samples"][checkpoint] = {
                "checkpoint": checkpoint,
                "duration": prompt["durations"].get(checkpoint) or wav_duration(path) or "-",
                "exists": path.exists(),
                "path": str(path),
                "url": sample_url(path),
                "eval": latest_eval.get(f"{prompt['id']}::{checkpoint}"),
            }
        rows.append(row)
    return rows


def build_infer_command(
    *,
    checkpoint: str,
    text: str,
    output_path: Path,
    seed: int = 42,
    speed: float = DEFAULT_SPEED,
) -> list[str]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    return [
        str(OMNIVOICE_PYTHON),
        "-c",
        (
            "from omnivoice.utils.common import fix_random_seed; "
            f"fix_random_seed({seed}); "
            "from omnivoice.cli.infer import main; main()"
        ),
        "--model",
        str(CHECKPOINT_ROOT / checkpoint),
        "--text",
        text,
        "--output",
        str(output_path),
        "--ref_audio",
        str(REF_AUDIO),
        "--ref_text",
        REF_TEXT,
        "--language",
        "en",
        "--num_step",
        "32",
        "--guidance_scale",
        "1.5",
        "--speed",
        f"{speed:g}",
    ]


def validate_runtime() -> None:
    required = [
        ("OmniVoice python", OMNIVOICE_PYTHON),
        ("OmniVoice repo", OMNIVOICE_ROOT),
        ("reference audio", REF_AUDIO),
    ]
    for checkpoint in CHECKPOINTS:
        required.append((checkpoint, CHECKPOINT_ROOT / checkpoint))
    missing = [f"{label}: {path}" for label, path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def append_history(record: GeneratedSample) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")


def read_history(limit: int = 30) -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with HISTORY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.reverse()
    return rows[:limit]


def _score(value: Any, label: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer from 1 to 5.") from exc
    if score < 1 or score > 5:
        raise ValueError(f"{label} must be an integer from 1 to 5.")
    return score


def normalize_eval_record(payload: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(payload.get("sample_id", "")).strip()
    checkpoint = str(payload.get("checkpoint", "")).strip()
    if not sample_id:
        raise ValueError("sample_id is required.")
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    notes = str(payload.get("notes", "")).strip()
    return {
        "created_at": iso_now(),
        "sample_id": sample_id,
        "checkpoint": checkpoint,
        "glados": _score(payload.get("glados"), "glados"),
        "clean": _score(payload.get("clean"), "clean"),
        "artifacts": _score(payload.get("artifacts"), "artifacts"),
        "notes": notes[:500],
    }


def write_eval_record(
    path: Path = EVAL_PATH,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = {}
    record = normalize_eval_record(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return record


def read_eval_records(path: Path = EVAL_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_eval_by_sample(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"{record['sample_id']}::{record['checkpoint']}"
        latest[key] = record
    return latest


def eval_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {checkpoint: [] for checkpoint in CHECKPOINTS}
    for record in records:
        checkpoint = record.get("checkpoint")
        if checkpoint in buckets:
            buckets[checkpoint].append(record)

    summary: dict[str, dict[str, Any]] = {}
    for checkpoint, rows in buckets.items():
        if not rows:
            summary[checkpoint] = {
                "count": 0,
                "glados_avg": None,
                "clean_avg": None,
                "artifacts_avg": None,
            }
            continue
        summary[checkpoint] = {
            "count": len(rows),
            "glados_avg": round(sum(row["glados"] for row in rows) / len(rows), 2),
            "clean_avg": round(sum(row["clean"] for row in rows) / len(rows), 2),
            "artifacts_avg": round(
                sum(row["artifacts"] for row in rows) / len(rows),
                2,
            ),
        }
    return summary


def parse_seed_list(value: Any) -> list[int]:
    if value in (None, ""):
        return DEFAULT_SEEDS
    if isinstance(value, str):
        raw_values = [part.strip() for part in re.split(r"[\s,]+", value) if part.strip()]
    elif isinstance(value, list):
        raw_values = value
    else:
        raise ValueError("Seeds must be a comma-separated string or list.")

    seeds: list[int] = []
    for raw_value in raw_values:
        seed = int(raw_value)
        if seed < 0 or seed > 2_147_483_647:
            raise ValueError("Seeds must be between 0 and 2147483647.")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        return DEFAULT_SEEDS
    if len(seeds) > MAX_SEEDS:
        raise ValueError(f"Use {MAX_SEEDS} seeds or fewer.")
    return seeds


def parse_checkpoint_list(value: Any) -> list[str]:
    if value in (None, "", "all"):
        return CHECKPOINTS
    if isinstance(value, str):
        checkpoints = [value]
    elif isinstance(value, list):
        checkpoints = [str(item) for item in value]
    else:
        raise ValueError("checkpoints must be a checkpoint name, list, or 'all'.")
    unknown = [checkpoint for checkpoint in checkpoints if checkpoint not in CHECKPOINTS]
    if unknown:
        raise ValueError("Unknown checkpoint(s): " + ", ".join(unknown))
    deduped = [checkpoint for checkpoint in CHECKPOINTS if checkpoint in set(checkpoints)]
    if not deduped:
        raise ValueError("Select at least one checkpoint.")
    return deduped


def parse_speed(value: Any) -> float:
    if value in (None, ""):
        return DEFAULT_SPEED
    speed = float(value)
    if speed < 0.5 or speed > 1.8:
        raise ValueError("Speed must be between 0.5 and 1.8.")
    return speed


def generate_for_checkpoints(
    text: str,
    *,
    checkpoints: list[str] | None = None,
    seeds: list[int] | None = None,
    speed: float = DEFAULT_SPEED,
) -> list[GeneratedSample]:
    validate_runtime()
    if checkpoints is None:
        checkpoints = CHECKPOINTS
    if seeds is None:
        seeds = DEFAULT_SEEDS
    job_count = len(checkpoints) * len(seeds)
    if job_count > MAX_GENERATION_JOBS:
        raise ValueError(f"Requested {job_count} jobs; limit is {MAX_GENERATION_JOBS}.")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(text)
    records: list[GeneratedSample] = []
    for checkpoint in CHECKPOINTS:
        if checkpoint not in checkpoints:
            continue
        for seed in seeds:
            output_path = GENERATED_DIR / f"{stamp}_{checkpoint}_seed-{seed}_{slug}.wav"
            command = build_infer_command(
                checkpoint=checkpoint,
                text=text,
                output_path=output_path,
                seed=seed,
                speed=speed,
            )
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(seed)
            start = time.time()
            subprocess.run(command, check=True, cwd=OMNIVOICE_ROOT, env=env)
            record = GeneratedSample(
                checkpoint=checkpoint,
                seed=seed,
                speed=speed,
                text=text,
                created_at=iso_now(),
                elapsed_seconds=round(time.time() - start, 2),
                path=str(output_path),
                url=sample_url(output_path),
            )
            append_history(record)
            records.append(record)
    return records


def generate_for_all_checkpoints(text: str) -> list[GeneratedSample]:
    return generate_for_checkpoints(text)


def app_payload() -> dict[str, Any]:
    eval_records = read_eval_records()
    return {
        "checkpoints": CHECKPOINTS,
        "checkpoint_meta": CHECKPOINT_META,
        "matrix": comparison_matrix(),
        "history": read_history(),
        "eval_summary": eval_summary(eval_records),
        "busy": STATE.current_job,
        "settings": {
            "num_step": 32,
            "guidance_scale": 1.5,
            "default_speed": DEFAULT_SPEED,
            "default_seeds": DEFAULT_SEEDS,
            "ref_audio": str(REF_AUDIO),
            "ref_text": REF_TEXT,
            "checkpoint_root": str(CHECKPOINT_ROOT),
        },
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OmniVoice Checkpoint Compare</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --border: #343d45;
      --text: #edf2f4;
      --muted: #aab4bd;
      --accent: #5eead4;
      --accent-2: #fbbf24;
      --danger: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      border-bottom: 1px solid var(--border);
      background: #14181b;
    }
    .wrap {
      width: min(1420px, calc(100vw - 32px));
      margin: 0 auto;
    }
    header .wrap {
      min-height: 76px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(24px, 3vw, 38px);
      line-height: 1.05;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      max-width: 660px;
    }
    main {
      padding: 20px 0 40px;
      display: grid;
      gap: 18px;
    }
    section {
      border-top: 1px solid var(--border);
      padding-top: 18px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    .table-scroll {
      overflow-x: auto;
    }
    th, td {
      border: 1px solid var(--border);
      padding: 12px;
      vertical-align: top;
      background: var(--panel);
    }
    th {
      background: var(--panel-2);
      color: var(--muted);
      text-align: left;
      font-weight: 650;
    }
    .prompt-cell {
      width: 28%;
      min-width: 260px;
    }
    .prompt-id {
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 7px;
      font-weight: 700;
    }
    .prompt-id span {
      color: var(--accent);
      font-size: 13px;
      text-transform: uppercase;
    }
    .prompt-text {
      color: var(--muted);
      font-size: 13px;
    }
    audio {
      width: 100%;
      min-width: 190px;
      height: 38px;
    }
    .duration {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .eval-panel {
      margin-top: 10px;
      display: grid;
      gap: 8px;
    }
    .scores {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    select, input {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 8px;
      background: #11161a;
      color: var(--text);
      font: inherit;
    }
    .save-rating {
      min-height: 34px;
      padding: 7px 10px;
      background: var(--panel-2);
      color: var(--text);
    }
    .saved-rating {
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .summary-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
    }
    .summary-card strong {
      display: block;
      color: var(--accent-2);
      margin-bottom: 8px;
    }
    .generator {
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 280px;
      gap: 16px;
      align-items: start;
    }
    .instructions {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
    }
    .instructions strong {
      display: block;
      color: var(--text);
      margin-bottom: 7px;
    }
    .instructions ul {
      margin: 0;
      padding-left: 18px;
    }
    .instructions li + li {
      margin-top: 5px;
    }
    .control-grid {
      display: grid;
      gap: 10px;
    }
    .checkpoint-options {
      display: grid;
      gap: 6px;
      padding: 8px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
    }
    .check-option {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 26px;
      color: var(--text);
      font-size: 13px;
    }
    .check-option input {
      width: 16px;
      min-height: 16px;
      margin: 0;
    }
    textarea {
      width: 100%;
      min-height: 124px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }
    button {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      min-height: 42px;
      background: var(--accent);
      color: #051312;
      font-weight: 750;
      cursor: pointer;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .side {
      display: grid;
      gap: 10px;
    }
    .status {
      min-height: 42px;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      color: var(--muted);
      background: var(--panel);
      overflow-wrap: anywhere;
    }
    .history-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 12px;
    }
    .history-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
    }
    .history-card strong {
      display: block;
      margin-bottom: 8px;
      color: var(--accent-2);
    }
    .history-card p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 12px;
    }
    .error {
      color: var(--danger);
    }
    @media (max-width: 900px) {
      header .wrap { align-items: flex-start; flex-direction: column; padding: 16px 0; }
      table { table-layout: auto; }
      .generator { grid-template-columns: 1fr; }
      .history-grid { grid-template-columns: 1fr; }
      .summary-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div>
        <h1>OmniVoice Checkpoint Compare</h1>
        <div class="meta">Audition the selected checkpoints with fixed reference audio, 32 diffusion steps, and guidance scale 1.5.</div>
      </div>
      <div class="meta" id="settings"></div>
    </div>
  </header>
  <main class="wrap">
    <section>
      <h2>Saved Comparison Samples</h2>
      <div id="summary"></div>
      <div id="matrix"></div>
    </section>
    <section>
      <h2>Generate New Comparison</h2>
      <div class="generator">
        <textarea id="prompt" maxlength="700">Please continue the test. I am almost certain the next result will be educational, assuming you survive long enough to appreciate it.</textarea>
        <div class="side">
          <div class="instructions">
            <strong>Prompt notes</strong>
            <ul>
              <li>Write the exact line you want spoken.</li>
              <li>Questions and punctuation are usually the most reliable style controls.</li>
              <li>Supported tags like [sigh] or [question-en] can be tested, but this fine-tune was not explicitly tagged for them.</li>
              <li>Seeds are real sampler seeds; use several to check whether flatness is stochastic or checkpoint-specific.</li>
            </ul>
          </div>
          <div class="control-grid">
            <label>Checkpoints<div class="checkpoint-options" id="checkpoint-options"></div></label>
            <label>Seeds<input id="seeds" value="42, 43, 44" inputmode="numeric"></label>
            <label>Speed<input id="speed" type="number" min="0.5" max="1.8" step="0.05" value="1.0"></label>
          </div>
          <button id="generate">Generate Across Selected Checkpoints</button>
          <div class="status" id="status">Ready.</div>
        </div>
      </div>
    </section>
    <section>
      <h2>Generated Samples</h2>
      <div id="history"></div>
    </section>
  </main>
  <script>
    let checkpoints = ["checkpoint-1500", "checkpoint-2000", "checkpoint-2500"];
    let checkpointMeta = [];
    const matrixEl = document.querySelector("#matrix");
    const summaryEl = document.querySelector("#summary");
    const historyEl = document.querySelector("#history");
    const statusEl = document.querySelector("#status");
    const generateBtn = document.querySelector("#generate");
    const promptEl = document.querySelector("#prompt");
    const checkpointOptionsEl = document.querySelector("#checkpoint-options");
    const seedsEl = document.querySelector("#seeds");
    const speedEl = document.querySelector("#speed");
    const settingsEl = document.querySelector("#settings");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function renderMatrix(rows) {
      const head = checkpoints.map((checkpoint) => `<th>${checkpoint}</th>`).join("");
      const body = rows.map((row) => {
        const cells = checkpoints.map((checkpoint) => {
          const sample = row.samples[checkpoint];
          if (!sample.exists) {
            return `<td class="error">Missing ${escapeHtml(sample.path)}</td>`;
          }
          const rating = sample.eval || {};
          return `<td>
            <audio controls preload="none" src="${sample.url}"></audio>
            <div class="duration">${sample.duration}</div>
            <div class="eval-panel" data-sample="${row.id}" data-checkpoint="${checkpoint}">
              <div class="scores">
                ${scoreSelect("glados", "GLaDOS", rating.glados)}
                ${scoreSelect("clean", "Clean", rating.clean)}
                ${scoreSelect("artifacts", "Artifacts", rating.artifacts)}
              </div>
              <input data-field="notes" value="${escapeHtml(rating.notes || "")}" placeholder="notes">
              <button class="save-rating" type="button">Save Rating</button>
              <div class="saved-rating">${rating.created_at ? "Saved" : ""}</div>
            </div>
          </td>`;
        }).join("");
        return `<tr>
          <td class="prompt-cell">
            <div class="prompt-id"><span>${row.id}</span>${escapeHtml(row.title)}</div>
            <div class="prompt-text">${escapeHtml(row.text)}</div>
          </td>
          ${cells}
        </tr>`;
      }).join("");
      matrixEl.innerHTML = `<div class="table-scroll"><table><thead><tr><th class="prompt-cell">Prompt</th>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function scoreSelect(field, label, selected) {
      const options = [1, 2, 3, 4, 5].map((value) => {
        const isSelected = Number(selected || 3) === value ? " selected" : "";
        return `<option value="${value}"${isSelected}>${value}</option>`;
      }).join("");
      return `<label>${label}<select data-field="${field}">${options}</select></label>`;
    }

    function renderSummary(summary) {
      summaryEl.innerHTML = `<div class="summary-grid">${checkpoints.map((checkpoint) => {
        const row = summary[checkpoint] || {};
        const meta = checkpointMeta.find((item) => item.name === checkpoint) || {};
        const evalText = meta.eval_loss === null || meta.eval_loss === undefined ? "" : `Eval ${meta.eval_loss}`;
        const best = meta.is_best_eval ? "Best eval" : "";
        const ratingText = row.count ? `GLaDOS ${row.glados_avg} / Clean ${row.clean_avg} / Artifacts ${row.artifacts_avg}` : "No ratings yet";
        const body = [best, evalText, ratingText].filter(Boolean).join(" | ");
        return `<div class="summary-card"><strong>${checkpoint}</strong><div class="meta">${body}</div></div>`;
      }).join("")}</div>`;
    }

    function renderHistory(rows) {
      if (!rows.length) {
        historyEl.innerHTML = `<div class="status">No generated samples yet.</div>`;
        return;
      }
      historyEl.innerHTML = `<div class="history-grid">${rows.map((row) => `
        <div class="history-card">
          <strong>${escapeHtml(row.checkpoint)}</strong>
          <audio controls preload="none" src="${row.url}"></audio>
          <p>${escapeHtml(row.text)}</p>
          <p>Seed ${escapeHtml(row.seed || "n/a")} | Speed ${escapeHtml(row.speed || 1)} | ${row.elapsed_seconds}s generation</p>
        </div>
      `).join("")}</div>`;
    }

    function renderCheckpointOptions() {
      const existing = new Set(
        Array.from(checkpointOptionsEl.querySelectorAll("input:checked")).map((input) => input.value)
      );
      const checked = existing.size ? existing : new Set(checkpoints);
      checkpointOptionsEl.innerHTML = checkpoints.map((checkpoint) => `
        <label class="check-option">
          <input type="checkbox" value="${checkpoint}" ${checked.has(checkpoint) ? "checked" : ""}>
          <span>${checkpoint}</span>
        </label>
      `).join("");
    }

    function selectedCheckpoints() {
      return Array.from(checkpointOptionsEl.querySelectorAll("input:checked")).map((input) => input.value);
    }

    async function loadState() {
      const response = await fetch("/api/state");
      const state = await response.json();
      checkpoints = state.checkpoints;
      checkpointMeta = state.checkpoint_meta || [];
      renderCheckpointOptions();
      renderSummary(state.eval_summary);
      renderMatrix(state.matrix);
      renderHistory(state.history);
      settingsEl.textContent = `Ref: ${state.settings.ref_audio}`;
      if (state.busy) {
        statusEl.textContent = state.busy.message;
        generateBtn.disabled = true;
      } else {
        generateBtn.disabled = false;
      }
    }

    async function saveRating(panel) {
      const payload = {
        sample_id: panel.dataset.sample,
        checkpoint: panel.dataset.checkpoint,
      };
      panel.querySelectorAll("[data-field]").forEach((field) => {
        payload[field.dataset.field] = field.value;
      });
      const response = await fetch("/api/eval", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || "Rating failed.");
      }
      panel.querySelector(".saved-rating").textContent = "Saved";
      await loadState();
    }

    async function generate() {
      const text = promptEl.value.trim();
      const selected = selectedCheckpoints();
      if (!text) {
        statusEl.innerHTML = `<span class="error">Enter text first.</span>`;
        return;
      }
      if (!selected.length) {
        statusEl.innerHTML = `<span class="error">Select at least one checkpoint.</span>`;
        return;
      }
      generateBtn.disabled = true;
      const seeds = seedsEl.value.trim();
      const speed = speedEl.value.trim();
      const seedCount = seeds ? seeds.split(/[,\\s]+/).filter(Boolean).length : 1;
      statusEl.textContent = `Generating ${selected.length * seedCount} samples.`;
      try {
        const response = await fetch("/api/generate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({text, checkpoints: selected, seeds, speed}),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Generation failed.");
        }
        statusEl.textContent = `Generated ${payload.records.length} samples.`;
        await loadState();
      } catch (error) {
        statusEl.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      } finally {
        generateBtn.disabled = false;
      }
    }

    generateBtn.addEventListener("click", generate);
    document.addEventListener("click", async (event) => {
      if (!event.target.classList.contains("save-rating")) {
        return;
      }
      const panel = event.target.closest(".eval-panel");
      event.target.disabled = true;
      try {
        await saveRating(panel);
      } catch (error) {
        panel.querySelector(".saved-rating").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      } finally {
        event.target.disabled = false;
      }
    });
    loadState().catch((error) => {
      statusEl.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
    });
  </script>
</body>
</html>
"""


class CompareHandler(BaseHTTPRequestHandler):
    server_version = "OmniCompare/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = 200,
        *,
        include_body: bool = True,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def send_json(
        self,
        payload: dict[str, Any],
        status: int = 200,
        *,
        include_body: bool = True,
    ) -> None:
        self.send_bytes(
            json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
            include_body=include_body,
        )

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:
        self.route_get(include_body=True)

    def do_HEAD(self) -> None:
        self.route_get(include_body=False)

    def route_get(self, *, include_body: bool) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(
                HTML.encode("utf-8"),
                "text/html; charset=utf-8",
                include_body=include_body,
            )
            return
        if path == "/api/state":
            self.send_json(app_payload(), include_body=include_body)
            return
        if path.startswith("/samples/"):
            self.serve_sample(path, include_body=include_body)
            return
        self.send_error_json("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/eval":
            self.handle_eval_post()
            return
        if path != "/api/generate":
            self.send_error_json("Not found", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", "")).strip()
            if not text:
                self.send_error_json("Text is required.")
                return
            if len(text) > MAX_TEXT_LEN:
                self.send_error_json(f"Text must be {MAX_TEXT_LEN} characters or fewer.")
                return
            checkpoints = parse_checkpoint_list(payload.get("checkpoints"))
            seeds = parse_seed_list(payload.get("seeds"))
            speed = parse_speed(payload.get("speed"))
            job_count = len(checkpoints) * len(seeds)
            if job_count > MAX_GENERATION_JOBS:
                self.send_error_json(
                    f"Requested {job_count} jobs; limit is {MAX_GENERATION_JOBS}."
                )
                return
            with STATE.lock:
                if STATE.current_job is not None:
                    self.send_error_json("A generation job is already running.", HTTPStatus.CONFLICT)
                    return
                STATE.current_job = {
                    "message": f"Generating {job_count} comparison samples.",
                    "text": text,
                    "checkpoints": checkpoints,
                    "seeds": seeds,
                    "speed": speed,
                }
            try:
                records = generate_for_checkpoints(
                    text,
                    checkpoints=checkpoints,
                    seeds=seeds,
                    speed=speed,
                )
            finally:
                with STATE.lock:
                    STATE.current_job = None
            self.send_json({"records": [asdict(record) for record in records]})
        except subprocess.CalledProcessError as exc:
            with STATE.lock:
                STATE.current_job = None
            self.send_error_json(f"OmniVoice failed with exit code {exc.returncode}.", 500)
        except Exception as exc:
            with STATE.lock:
                STATE.current_job = None
            self.send_error_json(str(exc), 500)

    def handle_eval_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            record = write_eval_record(payload=payload)
            self.send_json({"record": record})
        except Exception as exc:
            self.send_error_json(str(exc), 400)

    def serve_sample(self, request_path: str, *, include_body: bool) -> None:
        relative = unquote(request_path.removeprefix("/samples/"))
        path = (SAMPLE_ROOT / relative).resolve()
        try:
            path.relative_to(SAMPLE_ROOT.resolve())
        except ValueError:
            self.send_error_json("Invalid sample path.", HTTPStatus.FORBIDDEN)
            return
        if not path.exists() or path.suffix.lower() != ".wav":
            self.send_error_json("Sample not found.", HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not include_body:
            return
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 256):
                self.wfile.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OmniVoice checkpoint comparison UI.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), CompareHandler)
    print(f"Serving OmniVoice compare UI at http://{args.host}:{args.port}/")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
