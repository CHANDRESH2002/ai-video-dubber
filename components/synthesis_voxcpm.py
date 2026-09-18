"""
Synthesis -- VoxCPM2 voice cloning, with built-in natural-language style
control (a parenthetical instruction prepended to the text, e.g.
"(whispering, hesitant) मुझे यकीन नहीं हो रहा..."). Same interface/contract
as components/synthesis_chatterbox.py, offered as a second, parallel
engine -- see the integration plan for why this isn't a blind replacement
yet (needs a real end-to-end comparison first).

Picked over fine-tuning Chatterbox to add this capability: Chatterbox's T3
model has no conditioning slot for a style embedding at all (only two
scalar knobs, exaggeration/cfg_weight) -- adding one is architecture
surgery, not a fine-tune. VoxCPM2 (OpenBMB, Apache 2.0) already has this
trained in, and already speaks Hindi (unlike IndexTTS2, which has the same
style mechanism but no Hindi support and a non-permissive bilibili
license).

IMPORTANT: this module can only run inside the isolated `.venv_voxcpm`
environment -- voxcpm pulls its own torch (2.14.0 as installed), same
cross-venv reasoning as Chatterbox. Run via `.venv_voxcpm/bin/python3`.

Setup (fresh machine):
    python3 -m venv .venv_voxcpm
    .venv_voxcpm/bin/pip install voxcpm
Requires Python >=3.10,<3.13.
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
    Same contract as components/synthesis_chatterbox.py's synthesize().
    segments: must already have `speaker` set; `style_prompt` is optional
        (see components/style_prompt.py) -- when present, it's prepended
        as a parenthetical style instruction ahead of the spoken text.
    """
    import soundfile as sf
    from voxcpm import VoxCPM

    from components import voxcpm_alignment_patch
    from components.audio_trim import trim_trailing_silence
    from components.speed_adjust import fit_duration

    print("Loading VoxCPM2...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    voxcpm_alignment_patch.apply(model)
    sample_rate = model.tts_model.sample_rate

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

        style_prompt = getattr(seg, "style_prompt", "") or ""
        prompted_text = f"({style_prompt}){text}" if style_prompt.strip() else text

        already_done = out_path.exists()
        if already_done:
            data, sr = sf.read(str(out_path))
            actual_dur = len(data) / sr
        else:
            wav = model.generate(
                text=prompted_text,
                reference_wav_path=str(reference_audio),
                cfg_value=2.0,
                inference_timesteps=10,
                # VoxCPM2's own stop-token classifier occasionally fails to
                # fire (same category of bug as chatterbox_patch.py's T3
                # fixes) and generation runs out to a generous internal cap
                # instead -- retry_badcase defaults to False in the
                # installed package, so this check never ran until now. See
                # scratchpad_reports/voxcpm2_dataflow.html for the traced
                # mechanism. ratio_threshold tightened from the library's
                # default of 6.0 -- dubbing needs tight duration matching,
                # and 6x was letting through exactly the overflow we saw.
                retry_badcase=True,
                retry_badcase_ratio_threshold=3.0,
            )
            sf.write(str(out_path), wav, sample_rate)
            _, actual_dur = trim_trailing_silence(str(out_path), str(out_path))

        # Same last-resort bounded cleanup as the Chatterbox path -- see
        # components/synthesis_chatterbox.py for why this also runs on
        # already-done/resumed segments.
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
        style_note = f"  style=\"{style_prompt}\"" if style_prompt.strip() else ""
        print(f"  segment {i+1}/{len(segments)} [{seg.speaker}]: [{seg.start:.1f}s → {seg.end:.1f}s] "
              f"target={target_dur:.1f}s actual={actual_dur:.1f}s{flag}{skipped}{style_note}  \"{text[:50]}\"")

    return results
