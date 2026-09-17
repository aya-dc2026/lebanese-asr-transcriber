"""
asr_engine.py

Phase 1: a minimal, dependency-light wrapper around a Qwen3-ASR checkpoint that:
  - loads the model + processor once (call load_model() at startup)
  - normalizes arbitrary audio (mp3/mp4/wav/...) to 16kHz mono wav via ffmpeg
  - runs inference and returns transcript text (+ optional per-segment info)

Phase 2 will import `transcribe()` from this module inside the FastAPI app
instead of duplicating model-loading logic.

Usage (Phase 1 baseline CLI):
    python backend/asr_engine.py --audio path/to/clip.mp3 --model models/base/qwen3-asr-0.6b-arabic-dialectal
"""
import argparse
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TranscriptionResult:
    text: str
    model_path: str
    # Placeholder for future confidence/timestamp info (see HLD 7.2, 10)
    segments: Optional[list] = None


class ASREngine:
    """Loads a Qwen3-ASR checkpoint once and keeps it resident for repeated calls."""

    def __init__(self, model_path: str, device: Optional[str] = None, adapter_path: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        # NOTE: import here (not top-level) so this file can be inspected/linted
        # even before transformers/peft are installed.
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        print(f"[asr_engine] loading base model from {self.model_path} on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        if self.adapter_path:
            self._load_adapter(self.adapter_path)

    def _load_adapter(self, adapter_path: str):
        """Hot-swap a LoRA adapter onto the base model (Phase 4+)."""
        from peft import PeftModel

        print(f"[asr_engine] attaching LoRA adapter from {adapter_path} ...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.adapter_path = adapter_path

    def swap_adapter(self, adapter_path: str):
        """Call this after a retrain finishes, without restarting the process."""
        self._load_adapter(adapter_path)

    @staticmethod
    def normalize_audio(input_path: str) -> str:
        """Convert any input audio/video to 16kHz mono wav using ffmpeg.
        Returns path to a temp wav file the caller is responsible for cleaning up.
        """
        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1",
            out_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        import torchaudio

        wav_path = self.normalize_audio(audio_path)
        try:
            waveform, sr = torchaudio.load(wav_path)
            inputs = self.processor(
                waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=440)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return TranscriptionResult(text=text.strip(), model_path=self.model_path)
        finally:
            Path(wav_path).unlink(missing_ok=True)


def _cli():
    parser = argparse.ArgumentParser(description="Phase 1 baseline: transcribe a single audio file.")
    parser.add_argument("--audio", required=True, help="Path to mp3/mp4/wav file")
    parser.add_argument("--model", required=True, help="Path to a local model directory")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path (Phase 4+)")
    args = parser.parse_args()

    engine = ASREngine(model_path=args.model, adapter_path=args.adapter)
    result = engine.transcribe(args.audio)
    print("\n=== Transcript ===")
    print(result.text)


if __name__ == "__main__":
    _cli()
