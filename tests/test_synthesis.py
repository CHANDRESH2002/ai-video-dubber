"""
Standalone validation script for per-speaker synthesis — runs the full
transcribe -> diarize -> align -> synthesize chain on a real conversation
clip and reports target-vs-actual duration per segment (same signal we used
to validate duration-matching in pipeline.py originally).

Usage:
    HF_TOKEN=hf_xxx python tests/test_synthesis.py <vocals_path> <duration> [num_speakers]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN
from components.diarization import diarize, extract_speaker_audio
from components.transcription import transcribe
from components.alignment import assign_speakers
from components.synthesis import synthesize


def main():
    if len(sys.argv) < 3:
        print("Usage: python tests/test_synthesis.py <vocals_path> <duration> [num_speakers]")
        sys.exit(1)

    vocals_path = Path(sys.argv[1])
    duration = float(sys.argv[2])
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    print("Transcribing...")
    segments = transcribe(vocals_path, duration=duration)

    print("Diarizing (overlap-aware, for alignment)...")
    speaker_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers)

    print("Diarizing (exclusive, for clean reference clips)...")
    clean_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers, exclusive=True)

    print("Aligning...")
    tagged_segments = assign_speakers(segments, speaker_spans)

    speakers = sorted(set(s.speaker for s in clean_spans))
    print(f"Building reference clips for: {speakers}")
    ref_dir = Path("temp/speaker_references")
    ref_dir.mkdir(parents=True, exist_ok=True)
    speaker_references = {}
    for speaker in speakers:
        ref_path = ref_dir / f"{speaker}.wav"
        extract_speaker_audio(vocals_path, clean_spans, speaker, ref_path, max_duration=20.0)
        speaker_references[speaker] = ref_path

    print("\nSynthesizing...\n")
    output_dir = Path("temp/synthesis_test")
    results = synthesize(tagged_segments, speaker_references, output_dir, target_lang="en")

    total_target = sum(s.end - s.start for s in tagged_segments if s.text.strip())
    total_actual = sum(s.duration for s in results)
    print(f"\nTotal: target={total_target:.1f}s actual={total_actual:.1f}s "
          f"({(total_actual / total_target - 1) * 100:+.0f}%)")


if __name__ == "__main__":
    main()
