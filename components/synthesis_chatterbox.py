"""
Synthesis — Chatterbox Multilingual voice cloning, one reference clip per
speaker. Same interface/contract as components/synthesis.py (the XTTS
version), swapped to Chatterbox after a direct comparison on the same test
lines showed Chatterbox producing clearly better, more varied-sounding
speech (XTTS came out flat/robotic on 3 of 4 test lines; Chatterbox sounded
natural on all 4). MIT-licensed too, vs XTTS's non-commercial CPML license.

IMPORTANT: this module can only run inside the isolated `.venv_chatterbox`
environment (see /Users/chandreshpatel/dubbing/.venv_chatterbox) — the
chatterbox-tts package hard-pins torch==2.6.0/transformers==5.2.0, which
are incompatible with the main environment's pyannote/coqui-tts/IndicTrans2
stack. Run this file's logic via `.venv_chatterbox/bin/python3`, never the
main interpreter.

Setup (fresh machine, e.g. a rented GPU box):
    python3 -m venv .venv_chatterbox
    .venv_chatterbox/bin/pip install chatterbox-tts
    .venv_chatterbox/bin/pip install "setuptools<81"

That last line matters: chatterbox-tts depends on resemble-perth for audio
watermarking, which imports pkg_resources at import time. Newer setuptools
(observed: 84.0.0, on a fresh Ubuntu 24.04 box) dropped pkg_resources
entirely, and resemble-perth's own import is wrapped in a silent
try/except ImportError -- so instead of an import error, you get a
confusing `TypeError: 'NoneType' object is not callable` deep inside
`ChatterboxMultilingualTTS.from_pretrained()`, at the `perth.PerthImplicitWatermarker()`
call. Pinning setuptools<81 restores pkg_resources and fixes it.

Also on a fresh Linux box, watch for a container-level `PIP_CONSTRAINT` env
var (seen on at least one GPU rental platform) forcing an unrelated,
non-PyPI torch build -- `unset PIP_CONSTRAINT` before any pip install in
either venv if you hit a torch dependency-resolution error.
"""

from pathlib import Path

from data_types import Segment, SynthesizedSegment


def synthesize(
    segments: list[Segment],
    speaker_references: dict[str, Path],
    output_dir: Path,
    target_lang: str = "hi",
) -> list[SynthesizedSegment]:
    """
    segments: must already have `speaker` set (see components/alignment.py).
    speaker_references: maps speaker label (e.g. "SPEAKER_00") to a clean,
        non-overlapping reference audio clip for that speaker (see
        components/diarization.py's extract_speaker_audio(..., exclusive=True)).
        Segments whose speaker has no entry here fall back to whichever
        reference clip is first in this dict.
    output_dir: where synthesized per-segment WAVs get written.
    """
    import soundfile as sf
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    from config import best_device
    from components import chatterbox_patch
    from components.audio_trim import trim_trailing_silence
    from components.speed_adjust import fit_duration

    chatterbox_patch.apply()
    device = best_device()
    print(f"Loading Chatterbox Multilingual on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

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

        already_done = out_path.exists()
        if already_done:
            # Resuming after an earlier crash -- this segment was already
            # synthesized (and trimmed) last time, don't redo real GPU work.
            data, sr = sf.read(str(out_path))
            actual_dur = len(data) / sr
        else:
            wav = model.generate(text, language_id=target_lang, audio_prompt_path=str(reference_audio))
            ta.save(str(out_path), wav, model.sr)
            _, actual_dur = trim_trailing_silence(str(out_path), str(out_path))

        # Last-resort, bounded cleanup for whatever's still off-target after
        # generation + trimming -- also runs on resumed/already-done
        # segments (cheap, ffmpeg-only, no GPU) so re-running this on an
        # already-completed manifest still benefits without wasting the
        # GPU work that's already done. No-ops if already within target;
        # fit_duration handles its own scratch-file/replace internally.
        actual_dur = fit_duration(str(out_path), str(out_path), target_dur)

        results.append(SynthesizedSegment(
            start=seg.start,
            end=seg.end,
            duration=actual_dur,
            path=str(out_path),
            speaker=seg.speaker,
            text=text,
        ))

        overflow = actual_dur - target_dur
        flag = f"  ⚠ {overflow:.1f}s over target" if overflow > 0.15 else ""
        skipped = "  [already done]" if already_done else ""
        print(f"  segment {i+1}/{len(segments)} [{seg.speaker}]: [{seg.start:.1f}s → {seg.end:.1f}s] "
              f"target={target_dur:.1f}s actual={actual_dur:.1f}s{flag}{skipped}  \"{text[:50]}\"")

    return results
