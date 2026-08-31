"""
Full chain + metrics report. This doubles as a repeatable evaluation
harness — rerun after changing any component to get a quantitative
before/after comparison instead of just re-listening.

Usage:
    HF_TOKEN=hf_xxx python tests/test_evaluation.py <video_path> <vocals_path> <background_path> <output_path> <target_lang> [num_speakers]
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN
from components.diarization import diarize, extract_speaker_audio
from components.transcription import transcribe
from components.alignment import assign_speakers
from components.translation import translate_segments
from components.synthesis import synthesize
from components.assembly import assemble
from components.evaluation import (
    duration_match_metrics, load_speaker_embedder, speaker_similarity,
    load_translation_quality_model, translation_quality,
)


def main():
    if len(sys.argv) < 6:
        print("Usage: python tests/test_evaluation.py <video_path> <vocals_path> <background_path> <output_path> <target_lang> [num_speakers]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    vocals_path = Path(sys.argv[2])
    background_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4])
    target_lang = sys.argv[5]
    num_speakers = int(sys.argv[6]) if len(sys.argv) > 6 else None

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

    print("Diarizing (overlap-aware, for alignment + assembly)...")
    speaker_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers)

    print("Diarizing (exclusive, for clean reference clips)...")
    clean_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers, exclusive=True)

    print("Aligning...")
    tagged_segments = assign_speakers(segments, speaker_spans)

    print(f"Translating (LLM, whole-conversation context, target_lang={target_lang})...")
    # Keep the English source text around (source_texts) — CometKiwi needs
    # (source, translation) PAIRS to score quality, and translate_segments()
    # returns a new list rather than mutating tagged_segments in place.
    source_texts = [seg.text for seg in tagged_segments]
    if target_lang != "en":
        tagged_segments = translate_segments(tagged_segments, target_lang)

    speakers = sorted(set(s.speaker for s in clean_spans))
    print(f"Building reference clips for: {speakers}")
    ref_dir = Path("temp/speaker_references")
    ref_dir.mkdir(parents=True, exist_ok=True)
    speaker_references = {}
    for speaker in speakers:
        ref_path = ref_dir / f"{speaker}.wav"
        extract_speaker_audio(vocals_path, clean_spans, speaker, ref_path)
        speaker_references[speaker] = ref_path

    print(f"\nSynthesizing (target_lang={target_lang})...\n")
    synth_dir = Path("temp/synthesis_test")
    synthesized = synthesize(tagged_segments, speaker_references, synth_dir, target_lang=target_lang)

    print("\nAssembling...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assemble(
        synthesized_segments=synthesized,
        background_path=background_path,
        input_video_path=video_path,
        output_video_path=output_path,
        temp_dir=Path("temp"),
    )

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    dur_metrics = duration_match_metrics(tagged_segments, synthesized)
    print("\nDuration matching:")
    for k, v in dur_metrics.items():
        print(f"  {k}: {v}")

    print("\nSpeaker similarity (reference vs. synthesized, per segment):")
    embedder = load_speaker_embedder(HF_TOKEN)
    sims_by_speaker = {}
    for seg in synthesized:
        ref = speaker_references.get(seg.speaker)
        if ref is None:
            continue
        sim = speaker_similarity(ref, Path(seg.path), embedder)
        sims_by_speaker.setdefault(seg.speaker, []).append(sim)

    for speaker, sims in sorted(sims_by_speaker.items()):
        avg = sum(sims) / len(sims)
        print(f"  {speaker}: avg={avg:.3f}  min={min(sims):.3f}  max={max(sims):.3f}  (n={len(sims)})")

    print("\nTranslation quality (CometKiwi, reference-free):")
    if target_lang == "en":
        print("  skipped — source and \"translation\" are both English (Whisper task=translate IS the translation step here)")
    else:
        pairs = [
            {"src": src, "mt": seg.text}
            for src, seg in zip(source_texts, tagged_segments)
            if seg.text.strip()
        ]
        comet_model = load_translation_quality_model()
        scores = translation_quality(comet_model, pairs)
        for pair, score in zip(pairs, scores):
            flag = "  ⚠ low" if score < 0.6 else ""
            print(f"  {score:.3f}{flag}  src=\"{pair['src'][:40]}\"  mt=\"{pair['mt'][:40]}\"")
        print(f"  Average: {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
