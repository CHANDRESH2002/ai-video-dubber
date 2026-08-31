"""
Synthesis — XTTS-v2 zero-shot voice cloning, one reference clip per speaker.

Duration matching (speed adjustment + ffmpeg time-stretch) was removed —
on the Hindi test clip, average overflow was ~37%, meaning a lot of
segments needed real compression to hit the target window. That forced
speed sounded mechanical/rushed and was suspected of hurting perceived
delivery quality independent of the voice-cloning itself. Each segment now
synthesizes at XTTS's natural pace and keeps whatever duration that
produces — segments run long or short relative to the original timing
instead of being squeezed. components/assembly.py's per-speaker cursor
already prevents a speaker's own segments from overlapping each other when
one runs over, so this doesn't break, it just lets audio drift out of sync
with the source video's timing as segments compound. If that drift turns
out to be the bigger problem, duration matching will need to come back in
some form — measure the tradeoff before assuming this is strictly better.

Each segment is cloned from *its own speaker's* reference clip
(speaker_references), not one global reference for the whole file — see
components/diarization.py's extract_speaker_audio() for how those
per-speaker clips get built.
"""

from pathlib import Path

import soundfile as sf

from config import XTTS_LANG_MAP
from data_types import Segment, SynthesizedSegment


def synthesize(
    segments: list[Segment],
    speaker_references: dict[str, Path],
    output_dir: Path,
    target_lang: str = "en",
) -> list[SynthesizedSegment]:
    """
    segments: must already have `speaker` set (see components/alignment.py).
    speaker_references: maps speaker label (e.g. "SPEAKER_00") to a clean,
        non-overlapping reference audio clip for that speaker (see
        components/diarization.py's extract_speaker_audio(..., exclusive=True)).
        Segments whose speaker has no entry here fall back to whichever
        reference clip is first in this dict, so synthesis still runs, but
        that shouldn't happen if diarization covered the whole conversation.
    output_dir: where synthesized per-segment WAVs get written.
    """
    from TTS.api import TTS
    from config import best_device

    device = best_device()
    print(f"Loading XTTS-v2 on device: {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    xtts_lang = XTTS_LANG_MAP.get(target_lang, "en")
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback_reference = next(iter(speaker_references.values()))

    results = []

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue

        reference_audio = speaker_references.get(seg.speaker, fallback_reference)
        target_dur = seg.end - seg.start
        out_path = output_dir / f"seg_{i:04d}.wav"

        # Natural pace only — no speed adjustment, no time-stretch. See
        # module docstring for why duration matching was removed.
        tts.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio),
            language=xtts_lang,
            file_path=str(out_path)
        )
        actual_dur = sf.info(str(out_path)).duration

        results.append(SynthesizedSegment(
            start=seg.start,
            end=seg.end,
            duration=actual_dur,
            path=str(out_path),
            speaker=seg.speaker,
            text=text,
        ))

        overflow = actual_dur - target_dur
        flag = f"  ⚠ still {overflow:.1f}s over" if overflow > 0.15 else ""
        print(f"  segment {i+1}/{len(segments)} [{seg.speaker}]: [{seg.start:.1f}s → {seg.end:.1f}s] "
              f"target={target_dur:.1f}s actual={actual_dur:.1f}s{flag}  \"{text[:50]}\"")

    return results
