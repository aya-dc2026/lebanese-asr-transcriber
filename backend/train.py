"""
train.py — LoRA fine-tuning pipeline (Phase 4)

Not implemented yet. Will:
  1. Export unused correction rows to JSONL
  2. Load base model (or current adapter) + apply LoRA config (peft)
  3. Fine-tune 1-3 epochs
  4. Save new adapter to models/adapters/vN_lora/
  5. Run scripts/evaluate.py against held-out set before promoting to "current"

See HLD 7.7 for full spec.
"""
