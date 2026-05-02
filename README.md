# GLaDOS TTS Build Kit

Source-only pipeline for building a personal GLaDOS-style TTS model from local Portal and Portal 2 game files.

This repository intentionally does not include Valve game assets, extracted audio, transcripts, token shards, generated samples, or trained checkpoints. You provide the installed game files and run the pipeline locally.

## What Is Included

| Stage | Files |
|---|---|
| Extract Portal/Portal 2 audio | `scripts/01_extract_vpks.sh` |
| Convert Source MP3-in-WAV to PCM | `scripts/02_transcode_pcm.sh` |
| CohereX transcription and alignment | `scripts/03_transcribe.sh` |
| Portal Wiki ground-truth scrape | `scripts/05_scrape_wiki.py` |
| Reconcile, bucket, and filter clips | `scripts/06_reconcile.py` |
| Human review UI and apply step | `scripts/07_build_review_page.py`, `scripts/08_apply_review.py` |
| Build training manifests | `scripts/04_build_manifest.py`, `scripts/09_convert_manifest.py` |
| OmniVoice training and inference | `scripts/10_train_omnivoice.sh`, `scripts/11_omni_infer.sh` |
| Local evaluation UI | `scripts/12_omni_compare_server.py` |

## Prerequisites

- Portal and Portal 2 installed locally through Steam.
- `uv`.
- `ffmpeg` and `ffprobe`.
- `vpk` CLI: `uv tool install vpk`.
- CohereX cloned separately for transcription: `git clone https://github.com/Diffio-AI/CohereX.git ~/git/CohereX && cd ~/git/CohereX && uv sync`.
- OmniVoice cloned separately for training/inference.
- An NVIDIA GPU is strongly recommended for transcription and training.

## Pipeline

Set `STEAM_COMMON` if Steam is not under the Linux default:

```bash
STEAM_COMMON="$HOME/.local/share/Steam/steamapps/common" bash scripts/01_extract_vpks.sh
bash scripts/02_transcode_pcm.sh
COHEREX_DIR="$HOME/git/CohereX" bash scripts/03_transcribe.sh
uv run python scripts/05_scrape_wiki.py
uv run python scripts/06_reconcile.py
uv run python scripts/07_build_review_page.py
uv run python -m http.server -d data 8765
```

Open `http://127.0.0.1:8765/review.html`, export decisions, then apply them:

```bash
uv run python scripts/08_apply_review.py --decisions ~/Downloads/review_decisions.json
uv run python scripts/04_build_manifest.py
uv run python scripts/09_convert_manifest.py
```

## OmniVoice

`scripts/10_train_omnivoice.sh` has two stages:

- `STAGE=0 STOP_STAGE=0`: tokenize `data/omni_train.jsonl` into WebDataset shards under `data/omni/tokens`.
- `STAGE=1 STOP_STAGE=1`: train from `config/omnivoice_train_config.json` and `config/omnivoice_data_config.json`.

```bash
OMNI="$HOME/git/OmniVoice" \
STAGE=0 STOP_STAGE=1 \
bash scripts/10_train_omnivoice.sh
```

Generate fixed prompt samples:

```bash
OMNI="$HOME/git/OmniVoice" bash scripts/11_omni_infer.sh
```

Serve the local comparison UI:

```bash
OMNIVOICE_ROOT="$HOME/git/OmniVoice" uv run python scripts/12_omni_compare_server.py --host 127.0.0.1 --port 8771
```

## Data Policy

All generated work products are ignored under `data/`: extracted game audio, PCM clips, Cohere transcripts, scraped cache, review HTML, token shards, samples, logs, and checkpoints. Keep this repo source-only. Do not publish trained weights or extracted Valve audio.

## License Reality Check

GLaDOS audio and Portal assets belong to Valve Corporation. This repo is for a local, personal build pipeline. Publishing the pipeline is different from publishing derivative audio assets or trained model weights.
