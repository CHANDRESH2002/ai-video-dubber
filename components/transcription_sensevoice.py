"""
Transcription via SenseVoiceSmall -- replaces Whisper as of this session's
testing. On a real 5-minute video, Whisper's long-form decoding fell into a
repetition loop (13x "Yeah." in under a second, each an impossibly short
"segment" that drove one speaker's timeline 8.36s out of sync via the
per-speaker placement cursor in components/assembly.py). SenseVoiceSmall
was tested head-to-head on the identical audio -- the exact clip and the
full pipeline run -- and produced zero repetition artifacts, while also
running roughly 8-9x faster than real-time (measured: 34-37s for 300s of
audio) and, on that same clip, losing zero segments to unattributed
speakers (vs. 40% lost on a harder, longer video with Whisper).

Trade-off accepted knowingly: SenseVoiceSmall has no translate-to-English
mode the way Whisper's task="translate" does (confirmed directly against
its own GitHub README). This pipeline's source audio has always been
English, so this hasn't blocked anything -- but it does mean this
component can't be used as-is if the project ever needs to dub from
non-English source audio. components/transcription.py (Whisper) is kept
in place, unused by default, for that future case.

Must run inside its own isolated venv (.venv_sensevoice) -- funasr pulls
its own transformers/tokenizers versions, and this project has been burned
before installing a new model's dependencies into the shared main env
(the same reasoning behind Chatterbox and BandIt each getting their own
venv -- see components/synthesis_chatterbox.py).

Setup (already done on this Mac):
    python3 -m venv sensevoice-experiment/.venv_sensevoice
    sensevoice-experiment/.venv_sensevoice/bin/pip install funasr torch torchaudio modelscope

Model license: MIT for FunASR's own code; model weights under the FunASR
Model Open Source License Agreement, which the maintainers explicitly
confirm permits commercial use with attribution -- verified before
adopting this, given this project's commercial intent.
"""
import json
import subprocess
from pathlib import Path

from data_types import Segment, SpeakerSpan

ROOT = Path(__file__).resolve().parent.parent
SENSEVOICE_PY = ROOT / "sensevoice-experiment" / ".venv_sensevoice" / "bin" / "python3"
RUNNER_SCRIPT = ROOT / "tests" / "run_sensevoice_transcription.py"
PERSPEAKER_RUNNER_SCRIPT = ROOT / "tests" / "run_sensevoice_transcription_perspeaker.py"


def transcribe(audio_path: Path, duration: float | None = None, language: str | None = None) -> list[Segment]:
    """
    Same signature/contract as components/transcription.py's transcribe()
    so callers don't need to change. `duration` and `language` are accepted
    but currently unused (SenseVoiceSmall doesn't need the end-of-content
    hallucination filter Whisper needed, and language="en" is hardcoded in
    the runner script since that's the only source language this pipeline
    has ever needed).
    """
    out_path = Path("temp/sensevoice_segments.json")
    subprocess.run(
        [str(SENSEVOICE_PY), str(RUNNER_SCRIPT), str(audio_path), str(out_path)],
        check=True,
    )
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    return [Segment(start=s["start"], end=s["end"], text=s["text"]) for s in raw]


def transcribe_per_speaker(audio_path: Path, speaker_blocks: list[SpeakerSpan]) -> list[Segment]:
    """
    Transcribes each diarization-derived speaker block SEPARATELY, so a
    transcript segment can never straddle a real speaker change -- see
    components/alignment.py's build_speaker_blocks() for how blocks are
    built, and the multi-speaker-attribution fix plan for why the
    whole-file transcribe() + assign_speakers() overlap-join combo could
    silently bury a real speaker change inside one merged "sentence."

    Returns segments with `.speaker` already set correctly -- no separate
    alignment/overlap-join step needed afterward.
    """
    blocks_path = Path("temp/sensevoice_speaker_blocks.json")
    blocks_path.write_text(json.dumps([
        {"start": b.start, "end": b.end, "speaker": b.speaker} for b in speaker_blocks
    ]), encoding="utf-8")

    out_path = Path("temp/sensevoice_perspeaker_segments.json")
    subprocess.run(
        [str(SENSEVOICE_PY), str(PERSPEAKER_RUNNER_SCRIPT), str(audio_path), str(blocks_path), str(out_path)],
        check=True,
    )
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    return [Segment(start=s["start"], end=s["end"], text=s["text"], speaker=s["speaker"]) for s in raw]
