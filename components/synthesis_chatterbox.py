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
    source_texts: list[str] | None = None,
    speaker_genders: dict[str, str] | None = None,
) -> list[SynthesizedSegment]:
    """
    segments: must already have `speaker` set (see components/alignment.py).
    speaker_references: maps speaker label (e.g. "SPEAKER_00") to a clean,
        non-overlapping reference audio clip for that speaker (see
        components/diarization.py's extract_speaker_audio(..., exclusive=True)).
        Segments whose speaker has no entry here fall back to whichever
        reference clip is first in this dict.
    output_dir: where synthesized per-segment WAVs get written.
    source_texts: the PRE-translation English text for each segment, same
        order/length as `segments` (seg.text here is already the translated
        text). Optional -- when given, a segment that's still overflowing
        after the bounded tempo-stretch below gets condensed-and-retranslated
        (see components/duration_fit.py) instead of just left long. When
        omitted, behavior is unchanged from before (tempo-stretch only).
    speaker_genders: speaker label -> "male"/"female", for Hindi verb
        agreement on any condensed retranslation (see
        components/gender_agreement_hindi.py). Only used with source_texts.
    """
    import json
    import os
    import subprocess
    import sys

    import soundfile as sf
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    from config import best_device
    from components import chatterbox_patch
    from components.audio_trim import trim_trailing_silence
    from components.speed_adjust import fit_duration
    from components.duration_fit import MAX_ATTEMPTS, SHRINK_STEP, TOLERANCE_S

    CONDENSE_WORKER = Path(__file__).resolve().parent / "condense_retry_worker.py"
    # On this Mac, the interpreter this very process is already running
    # under has both stacks installed (see CLAUDE.md's uname fallback) --
    # only a real Linux box needs MAIN_PY pointed at the separate main-env
    # venv. dub_multispeaker.sh exports MAIN_PY for that case.
    condense_py = os.environ.get("MAIN_PY") or sys.executable

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

        # Bounded, cheap cleanup first -- pitch-preserving tempo stretch,
        # capped at 1.3x either direction. Also runs on resumed/already-done
        # segments (ffmpeg-only, no GPU) so re-running this on an
        # already-completed manifest still benefits. No-ops if already
        # within target; fit_duration handles its own scratch-file/replace
        # internally.
        actual_dur = fit_duration(str(out_path), str(out_path), target_dur)
        overflow = actual_dur - target_dur

        # If the tempo cap couldn't close the gap, fall back to condensing
        # the ENGLISH source and retranslating -- shortens what's being
        # said instead of just squeezing its delivery (see
        # components/duration_fit.py). Only for freshly-synthesized
        # segments (not resumed -- don't reopen GPU work already committed
        # on a previous run) and only when source_texts was supplied.
        if not already_done and overflow > TOLERANCE_S and source_texts is not None:
            english_text = source_texts[i]
            gender = (speaker_genders or {}).get(seg.speaker)
            candidates = [(out_path, actual_dur, overflow, text)]
            current_english = english_text

            for attempt in range(MAX_ATTEMPTS - 1):
                try:
                    resp = subprocess.run(
                        [condense_py, str(CONDENSE_WORKER)],
                        input=json.dumps({
                            "text": current_english,
                            "shrink_fraction": SHRINK_STEP * (attempt + 1),
                            "target_lang": target_lang,
                            "gender": gender,
                        }),
                        capture_output=True, text=True, check=True,
                    )
                    result = json.loads(resp.stdout)
                except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
                    print(f"    condense-retry worker failed, keeping best so far: {e}")
                    break

                current_english = result["english"]
                candidate_text = result["translated"]
                candidate_path = output_dir / f"seg_{i:04d}_c{attempt}.wav"
                wav = model.generate(candidate_text, language_id=target_lang, audio_prompt_path=str(reference_audio))
                ta.save(str(candidate_path), wav, model.sr)
                _, candidate_dur = trim_trailing_silence(str(candidate_path), str(candidate_path))
                candidate_overflow = candidate_dur - target_dur
                candidates.append((candidate_path, candidate_dur, candidate_overflow, candidate_text))
                print(f"    condense retry {attempt + 1}/{MAX_ATTEMPTS - 1}: "
                      f"{candidate_dur:.1f}s (target {target_dur:.1f}s)  \"{candidate_text[:40]}\"")

                if candidate_overflow <= TOLERANCE_S:
                    break

            best_path, actual_dur, overflow, text = min(candidates, key=lambda c: abs(c[2]))
            if best_path != out_path:
                best_path.replace(out_path)
            for cand_path, *_ in candidates:
                if cand_path not in (out_path, best_path):
                    cand_path.unlink(missing_ok=True)

            # Final bounded polish on whichever candidate won.
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
