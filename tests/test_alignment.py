"""
Standalone validation script for transcription + speaker alignment — run
against the real conversation clip and eyeball whether speaker labels land
on the right lines.

Usage:
    HF_TOKEN=hf_xxx python tests/test_alignment.py <vocals_path> <duration> [num_speakers]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN
from components.diarization import diarize
from components.transcription import transcribe
from components.alignment import assign_speakers


def main():
    if len(sys.argv) < 3:
        print("Usage: python tests/test_alignment.py <vocals_path> <duration> [num_speakers]")
        sys.exit(1)

    vocals_path = Path(sys.argv[1])
    duration = float(sys.argv[2])
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    print("Transcribing...")
    segments = transcribe(vocals_path, duration=duration)

    print("Diarizing...")
    speaker_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers)

    print("Aligning...\n")
    tagged = assign_speakers(segments, speaker_spans)

    for seg in tagged:
        speaker = seg.speaker or "UNKNOWN"
        print(f"  [{seg.start:6.2f}s - {seg.end:6.2f}s]  {speaker:12s}  {seg.text}")


if __name__ == "__main__":
    main()
