"""
Stage 1/3 of the multi-speaker Chatterbox dub pipeline. Runs in the MAIN
environment (diarization/transcription/translation all live there).

Diarizes, transcribes, aligns, and translates, then builds per-speaker
reference clips and writes everything Chatterbox needs to a JSON manifest
for stage 2 (which runs in the isolated .venv_chatterbox environment,
since chatterbox-tts can't coexist with pyannote/coqui-tts/IndicTrans2 in
one environment -- see components/synthesis_chatterbox.py).

Usage:
    python tests/dub_multispeaker_prepare.py <video_path> <vocals_path> <target_lang> [num_speakers]
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN
from components.diarization import diarize, extract_speaker_audio
from components.transcription import transcribe
from components.alignment import assign_speakers
from components.translation import translate_segments


def main():
    if len(sys.argv) < 4:
        print("Usage: python tests/dub_multispeaker_prepare.py <video_path> <vocals_path> <target_lang> [num_speakers]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    vocals_path = Path(sys.argv[2])
    target_lang = sys.argv[3]
    num_speakers = int(sys.argv[4]) if len(sys.argv) > 4 else None

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())

    print("Transcribing...")
    segments = transcribe(vocals_path, duration=duration)

    print("Diarizing (overlap-aware, for alignment)...")
    speaker_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers)

    print("Diarizing (exclusive, for clean reference clips)...")
    clean_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers, exclusive=True)

    print("Aligning...")
    tagged_segments = assign_speakers(segments, speaker_spans)

    print(f"Translating (target_lang={target_lang})...")
    source_texts = [seg.text for seg in tagged_segments]
    if target_lang != "en":
        tagged_segments = translate_segments(tagged_segments, target_lang)

    speakers = sorted(set(s.speaker for s in clean_spans))
    print(f"Building reference clips for: {speakers}")
    ref_dir = Path("temp/speaker_references_multispeaker")
    ref_dir.mkdir(parents=True, exist_ok=True)
    speaker_references = {}
    for speaker in speakers:
        ref_path = ref_dir / f"{speaker}.wav"
        extract_speaker_audio(vocals_path, clean_spans, speaker, ref_path, max_duration=20.0)
        speaker_references[speaker] = str(ref_path)

    manifest = {
        "target_lang": target_lang,
        "video_path": str(video_path),
        "segments": [dataclasses.asdict(s) for s in tagged_segments],
        "source_texts": source_texts,
        "speaker_references": speaker_references,
    }
    out_path = Path("temp/multispeaker_manifest.json")
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(tagged_segments)} segments, {len(speakers)} speakers)")


if __name__ == "__main__":
    main()
