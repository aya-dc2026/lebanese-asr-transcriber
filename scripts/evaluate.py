"""
evaluate.py

Benchmarks one or more models/adapters against a fixed held-out test set,
so you can (a) pick your Phase 1 baseline and (b) confirm each retrain in
Phase 4 actually improves WER/CER before promoting it to "current".

Test set format (JSONL, one object per line):
    {"audio": "data/testset/clip001.wav", "text": "correct reference transcript"}

Usage:
    python scripts/evaluate.py \
        --model models/base/qwen3-asr-0.6b-arabic-dialectal \
        --testset data/testset/holdout.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

# allow "python scripts/evaluate.py" to import backend/asr_engine.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.asr_engine import ASREngine


def load_testset(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--testset", required=True, help="JSONL file: {audio, text} per line")
    args = parser.parse_args()

    import jiwer

    rows = load_testset(args.testset)
    if not rows:
        print(f"No rows found in {args.testset}")
        return

    engine = ASREngine(model_path=args.model, adapter_path=args.adapter)

    refs, hyps = [], []
    for row in rows:
        result = engine.transcribe(row["audio"])
        refs.append(row["text"])
        hyps.append(result.text)
        print(f"REF: {row['text']}")
        print(f"HYP: {result.text}")
        print("-" * 40)

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)

    print(f"\nModel:   {args.model}")
    print(f"Adapter: {args.adapter or '(none)'}")
    print(f"Samples: {len(rows)}")
    print(f"WER:     {wer:.4f}")
    print(f"CER:     {cer:.4f}")


if __name__ == "__main__":
    main()
