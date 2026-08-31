"""
End-to-end test: runs the full component chain (diarize -> transcribe ->
align -> synthesize -> assemble) on the conversation clip and produces a
final, watchable dubbed video.

Usage:
    HF_TOKEN=hf_xxx python tests/test_assembly.py <video_path> <vocals_path> <background_path> <output_path> [num_speakers]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN
from components.diarization import diarize, extract_speaker_audio
from components.transcription import transcribe
from components.alignment import assign_speakers
from components.synthesis import synthesize
from components.assembly import assemble


def main():
    if len(sys.argv) < 5:
        print("Usage: python tests/test_assembly.py <video_path> <vocals_path> <background_path> <output_path> [num_speakers]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    vocals_path = Path(sys.argv[2])
    background_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4])
    num_speakers = int(sys.argv[5]) if len(sys.argv) > 5 else None

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    import subprocess
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())

    print("Transcribing...")
    segments = transcribe(vocals_path, duration=duration)

    print("Diarizing (overlap-aware, for alignment + assembly)...")
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
    synth_dir = Path("temp/synthesis_test")
    synthesized = synthesize(tagged_segments, speaker_references, synth_dir, target_lang="en")

    print("\nAssembling...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assemble(
        synthesized_segments=synthesized,
        background_path=background_path,
        input_video_path=video_path,
        output_video_path=output_path,
        temp_dir=Path("temp"),
    )


if __name__ == "__main__":
    main()
