# Lebanese Arabic ASR Transcriber

Fully local speech-to-text for Lebanese/Syrian Arabic (with English/French
code-switching), built on Qwen3-ASR, with a human-correction loop that
feeds LoRA retraining. Full design in `docs/HLD.pdf`.

This README covers **Phase 0 (setup)** and **Phase 1 (baseline inference)**.
Later phases (web app, corrections, retraining) are stubbed out in
`backend/` and `frontend/` and will be filled in as you progress.

## Phase 0 — Setup

1. **GPU driver + CUDA.** Confirm your NVIDIA driver supports CUDA 12.x and
   the RTX 5090 (Blackwell / `sm_120`):
   ```bash
   nvidia-smi
   ```
   If PyTorch doesn't detect your GPU later, your driver/CUDA build is
   probably too old for Blackwell — update it first.

2. **Python environment** (3.11 or 3.12):
   ```bash
   cd lebanese-asr-transcriber
   python3 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install PyTorch with Blackwell (sm_120) support first**, before the rest
   of requirements.txt — check https://pytorch.org for the current install
   command for your CUDA version, e.g.:
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install the rest of the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **ffmpeg** (used to normalize mp3/mp4 → 16kHz mono wav):
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   # macOS
   brew install ffmpeg
   ```

6. **Set up secrets:**
   ```bash
   cp .env.example .env
   # edit .env: set MODEL_PATH once you've downloaded a model below,
   # and generate RETRAIN_PASSWORD_HASH when you get to Phase 4.
   ```

7. **GitHub repo:**
   ```bash
   git init
   git add .
   git commit -m "Phase 0: project scaffold"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
   (`.gitignore` already excludes model weights, uploads, and `.env`.)

## Phase 1 — Baseline inference (no training yet)

1. **Download model weights** (needs internet, one time only):
   ```bash
   chmod +x scripts/download_model.sh
   ./scripts/download_model.sh
   ```
   This pulls the vanilla `Qwen/Qwen3-ASR-0.6B` and the community Levantine
   dialectal fine-tune `oddadmix/qwen3-asr-0.6b-arabic-dialectal` into
   `models/base/`. Edit the script to also grab the 1.7B or Whisper variants
   if you want to benchmark those too.

2. **Transcribe a single clip from the command line:**
   ```bash
   python backend/asr_engine.py \
     --audio /path/to/your/clip.mp3 \
     --model models/base/qwen3-asr-0.6b-arabic-dialectal
   ```

3. **Benchmark base vs. dialectal fine-tune on your own clips.** Gather ~20
   short Lebanese/Syrian clips (WhatsApp voice notes are a good real-world
   sample), transcribe each with both models, and eyeball which is more
   accurate on your dialect and code-switching. If you have (or write)
   reference transcripts for a few of them, use the evaluator instead of
   eyeballing:
   ```bash
   # data/testset/holdout.jsonl — one JSON object per line:
   # {"audio": "data/testset/clip001.wav", "text": "reference transcript"}
   python scripts/evaluate.py \
     --model models/base/qwen3-asr-0.6b-arabic-dialectal \
     --testset data/testset/holdout.jsonl
   ```
   Whichever model scores lower WER/CER (or sounds better to you) becomes
   your baseline — set it as `MODEL_PATH` in `.env`.

4. **Decide script vs. Arabizi output now** (HLD §7.8) — it's much easier to
   settle this before you build the correction UI than to switch later.

Once you're happy with a baseline, move on to Phase 2 (`backend/main.py`,
`frontend/`) to wrap this in a local web app.

## Repository layout

```
lebanese-asr-transcriber/
├── models/base/          model weights (git-ignored — see download_model.sh)
├── models/adapters/      LoRA adapters produced by retraining (Phase 4+)
├── data/                 uploads, corrections.db, training exports
├── backend/              asr_engine.py (Phase 1, done), main.py/db.py/auth.py/train.py (stubs)
├── frontend/             local web UI (Phase 2, stub)
├── scripts/              download_model.sh, evaluate.py
└── docs/                 this HLD report
```

## Status

- [x] Phase 0 — project scaffold
- [x] Phase 1 — baseline CLI transcription + evaluator (`backend/asr_engine.py`, `scripts/evaluate.py`)
- [ ] Phase 2 — FastAPI + minimal web UI
- [ ] Phase 3 — correction loop + SQLite storage
- [ ] Phase 4 — LoRA retraining pipeline
- [ ] Phase 5 — hardening & polish
