"""
Standalone validation script for the diarization component — run it directly
against a real audio file and eyeball whether the output plausibly matches
reality. Not a unit test with mocks: diarization quality can only be judged
against real audio, per the "validate each component on its own" approach.

Usage:
    HF_TOKEN=hf_xxx python tests/test_diarization.py <audio_path> [num_speakers]
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN
from components.diarization import diarize


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/test_diarization.py <audio_path> [num_speakers]")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("See the plan for setup steps (HuggingFace account + pyannote license + token).")
        sys.exit(1)

    if not audio_path.exists():
        print(f"ERROR: {audio_path} does not exist.")
        sys.exit(1)

    print(f"Running diarization on {audio_path} (num_speakers={num_speakers or 'auto'})...")
    spans = diarize(audio_path, hf_token=HF_TOKEN, num_speakers=num_speakers)

    print(f"\n{len(spans)} spans found:\n")
    for s in spans:
        print(f"  {s.start:6.2f}s - {s.end:6.2f}s   {s.speaker}")

    speaker_time = defaultdict(float)
    for s in spans:
        speaker_time[s.speaker] += s.end - s.start

    print(f"\n{len(speaker_time)} distinct speakers:")
    for speaker, total in sorted(speaker_time.items()):
        print(f"  {speaker}: {total:.1f}s total speaking time")


if __name__ == "__main__":
    main()
