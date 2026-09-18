"""
EXPERIMENTAL: transcription via Qwen3-ASR-1.7B instead of SenseVoice/Whisper.

On action.mp4 (a noisy, fast, multi-speaker action-set clip), Qwen3-ASR
recovered the entire opening exchange that both Whisper and SenseVoice
missed, and correctly recognized "Space cowboy ain't scared no ravine"
where SenseVoice produced garbled text ("Sp cowboy ain't scared"). See
scratchpad_reports/asr_comparison_action.txt for the full before/after.

Qwen3-ASR itself returns one continuous text block with no timestamps, so
this chains two isolated venvs: FunASR's standalone VAD (fsmn-vad, already
installed in .venv_sensevoice) finds {start, end} speech boundaries first
-- coarse ~8s chunks aligned to detected speech, not true per-sentence
segmentation -- then Qwen3-ASR (its own .venv_qwen3asr, needs
transformers>=5.13.0) transcribes each chunk. Each isolated venv runs as
its own subprocess, same reasoning as every other cross-venv split in this
project (see components/synthesis_chatterbox.py).

Setup (fresh machine):
    python3 -m venv qwen3-asr-experiment/.venv_qwen3asr
    qwen3-asr-experiment/.venv_qwen3asr/bin/pip install "transformers>=5.13.0" torch torchaudio accelerate librosa soundfile
Also requires sensevoice-experiment/.venv_sensevoice for the VAD step --
see components/transcription_sensevoice.py's own Setup block.
"""
import json
import subprocess
from pathlib import Path

from data_types import Segment

ROOT = Path(__file__).resolve().parent.parent
SENSEVOICE_PY = ROOT / "sensevoice-experiment" / ".venv_sensevoice" / "bin" / "python3"
QWEN3ASR_PY = ROOT / "qwen3-asr-experiment" / ".venv_qwen3asr" / "bin" / "python3"


def transcribe(audio_path: Path, duration: float | None = None, language: str | None = None) -> list[Segment]:
    """Same signature/contract as the other transcription components."""
    vad_out = Path("temp/qwen3asr_vad_segments.json")
    subprocess.run(
        [str(SENSEVOICE_PY), "tests/run_vad_segments.py", str(audio_path), str(vad_out)],
        check=True,
    )

    segments_out = Path("temp/qwen3asr_segments.json")
    subprocess.run(
        [str(QWEN3ASR_PY), "tests/run_qwen3asr_transcription.py", str(audio_path), str(vad_out), str(segments_out)],
        check=True,
    )

    raw = json.loads(segments_out.read_text(encoding="utf-8"))
    return [Segment(start=s["start"], end=s["end"], text=s["text"]) for s in raw]
