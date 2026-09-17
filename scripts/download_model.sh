#!/usr/bin/env bash
# Pulls model weights from Hugging Face into models/base/.
# Run this ONCE while you have internet access. Everything after this is offline.
#
# Usage:
#   chmod +x scripts/download_model.sh
#   ./scripts/download_model.sh

set -euo pipefail

MODELS_DIR="$(dirname "$0")/../models/base"
mkdir -p "$MODELS_DIR"

echo "== Installing huggingface_hub CLI if needed =="
pip install -q --upgrade huggingface_hub

download() {
  local repo_id="$1"
  local local_dir="$2"
  echo ""
  echo "== Downloading $repo_id -> $local_dir =="
  huggingface-cli download "$repo_id" \
    --local-dir "$local_dir" \
    --local-dir-use-symlinks False
}

# 1. Vanilla base models (pick one size to start; 0.6B is fastest to iterate on)
download "Qwen/Qwen3-ASR-0.6B" "$MODELS_DIR/qwen3-asr-0.6b"
# download "Qwen/Qwen3-ASR-1.7B" "$MODELS_DIR/qwen3-asr-1.7b"

# 2. Community Levantine-dialect fine-tune (recommended starting checkpoint)
download "oddadmix/qwen3-asr-0.6b-arabic-dialectal" "$MODELS_DIR/qwen3-asr-0.6b-arabic-dialectal"

# 3. Optional Whisper fallback/dialectal fine-tune
# download "oddadmix/whisper-large-v3-turbo-arabic-dialectal" "$MODELS_DIR/whisper-large-v3-turbo-arabic-dialectal"

echo ""
echo "Done. Verify sizes with: du -sh $MODELS_DIR/*"
echo "Next: run scripts/evaluate.py or backend/asr_engine.py to benchmark on your own clips."
