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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN
from components.diarization import diarize, extract_speaker_audio
from components.transcription_sensevoice import transcribe_per_speaker
from components.alignment import build_speaker_blocks
from components.translation import translate_segments
from components.speaker_gender import classify_speaker_gender
from components.gender_agreement_hindi import correct_gender_agreement
from components.style_prompt import get_style_prompt
from components.number_localization import protect_numbers, restore_numbers


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

    # Diarize FIRST, then transcribe separately within each speaker's own
    # turns -- guarantees a transcript segment can never straddle a real
    # speaker change (SenseVoice splits sentences on punctuation, not on
    # who's talking, so transcribing the whole file up front could -- and
    # on real test clips, did -- silently merge two different speakers'
    # dialogue into one "sentence," with no way to recover which part
    # belonged to whom afterward). Only one diarization pass needed now:
    # the old second overlap-aware pass existed solely to feed the
    # overlap-join in assign_speakers(), which this replaces.
    print("Diarizing (exclusive, for speaker blocks + clean reference clips)...")
    clean_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers, exclusive=True)

    print("Building speaker blocks...")
    speaker_blocks = build_speaker_blocks(clean_spans)

    print("Transcribing per speaker (SenseVoiceSmall)...")
    tagged_segments = transcribe_per_speaker(vocals_path, speaker_blocks)

    # Style prompts are computed from the ORIGINAL English audio -- the
    # real performance (whisper, anger, pace) lives in how the source
    # speaker actually said it, independent of target language. Must run
    # before translation replaces seg.text with the Hindi rendering.
    print("Generating style prompts (from source English audio)...")
    styled_segments = []
    for seg in tagged_segments:
        if seg.text.strip():
            prompt = get_style_prompt(vocals_path, seg.start, seg.end, seg.text)
            seg = dataclasses.replace(seg, style_prompt=prompt)
            print(f"  [{seg.speaker}] {seg.start:.1f}-{seg.end:.1f}s: \"{prompt}\"")
        styled_segments.append(seg)
    tagged_segments = styled_segments

    speakers = sorted(set(s.speaker for s in clean_spans))
    print(f"Building reference clips for: {speakers}")
    ref_dir = Path("temp/speaker_references_multispeaker")
    ref_dir.mkdir(parents=True, exist_ok=True)
    speaker_references = {}
    for speaker in speakers:
        ref_path = ref_dir / f"{speaker}.wav"
        extract_speaker_audio(vocals_path, clean_spans, speaker, ref_path, max_duration=20.0)
        speaker_references[speaker] = str(ref_path)

    # Classify each speaker's gender from their own reference clip -- the
    # English source text carries none (see components/speaker_gender.py),
    # so IndicTrans2 has no way to get first-person Hindi verb agreement
    # right without this. Done once per speaker here, before translation,
    # so the result is ready to apply right after.
    print("Classifying speaker gender (for Hindi verb agreement)...")
    speaker_genders = {
        speaker: classify_speaker_gender(Path(ref_path))
        for speaker, ref_path in speaker_references.items()
    }
    print(f"  {speaker_genders}")

    print(f"Translating (target_lang={target_lang})...")
    source_texts = [seg.text for seg in tagged_segments]
    if target_lang != "en":
        # Numbers are kept in spoken English rather than translated --
        # IndicTrans2 just copies digit sequences through untranslated
        # ("178,000" stays "178,000", unpronounceable by TTS), and real
        # Hindi speech commonly says numbers in English mid-sentence
        # anyway. Protect each segment's digits as numeric placeholders
        # (IndicTrans2 reliably leaves pure digit runs alone -- that's
        # exactly the behavior being exploited here) before translating,
        # then restore them as English words after. See
        # components/number_localization.py for why a letter-based
        # placeholder doesn't work (IndicTrans2 can phonetically
        # transliterate it instead of leaving it untouched).
        placeholder_maps = []
        protected_segments = []
        for seg in tagged_segments:
            protected_text, placeholders = protect_numbers(seg.text)
            placeholder_maps.append(placeholders)
            protected_segments.append(dataclasses.replace(seg, text=protected_text))

        translated_segments = translate_segments(protected_segments, target_lang)

        tagged_segments = [
            dataclasses.replace(seg, text=restore_numbers(seg.text, placeholders))
            for seg, placeholders in zip(translated_segments, placeholder_maps)
        ]

    if target_lang == "hi":
        print("Correcting Hindi first-person gender agreement...")
        corrected_segments = []
        for seg in tagged_segments:
            gender = speaker_genders.get(seg.speaker, "unknown")
            if gender in ("male", "female") and seg.text.strip():
                result = correct_gender_agreement(seg.text, gender)
                seg = dataclasses.replace(seg, text=result["text"])
            corrected_segments.append(seg)
        tagged_segments = corrected_segments

    manifest = {
        "target_lang": target_lang,
        "video_path": str(video_path),
        "segments": [dataclasses.asdict(s) for s in tagged_segments],
        "source_texts": source_texts,
        "speaker_references": speaker_references,
        "speaker_genders": speaker_genders,
    }
    out_path = Path("temp/multispeaker_manifest.json")
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(tagged_segments)} segments, {len(speakers)} speakers)")


if __name__ == "__main__":
    main()
