"""
Local, single-process test runner for the duration-fit + bidirectional
speed-adjust improvement -- Mac only. Unlike dub_multispeaker.sh's 3-stage,
2-environment production flow, this needs no cross-venv boundary: on this
Mac's pyenv interpreter, pyannote-audio and chatterbox-tts already coexist
in the same process (verified directly -- this is NOT true on the Linux GPU
box, where the venv split in CLAUDE.md's Environments section is real and
load-bearing; do not port this script there without rebuilding that split).

Runs the whole pipeline -- diarize, transcribe, align, translate-with-
duration-fit, synthesize, speed-adjust, assemble -- on one short clip so the
fit/retry loop can be iterated on quickly, then prints before/after duration-
match metrics so the improvement is measurable, not just assumed.

Usage:
    /Users/chandreshpatel/.pyenv/versions/3.12.5/bin/python3 \\
        tests/local_duration_fit_run.py <video_path> [target_lang] [num_speakers]
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN, best_device
from data_types import Segment, SynthesizedSegment
from components.diarization import diarize, extract_speaker_audio
from components.transcription import transcribe
from components.alignment import assign_speakers
from components.translation import translate_one
from components.duration_fit import fit_segment
from components.speed_adjust import fit_duration
from components.audio_trim import trim_trailing_silence
from components.assembly import assemble
from components.evaluation import duration_match_metrics, load_speaker_embedder, speaker_similarity
from components.subtitles import write_srt, burn_subtitles


def main():
    if len(sys.argv) < 2:
        print("Usage: local_duration_fit_run.py <video_path> [target_lang] [num_speakers]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    target_lang = sys.argv[2] if len(sys.argv) > 2 else "hi"
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    temp_dir = Path("temp/local_fit_run")
    temp_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())

    print("[1/7] Extracting audio...")
    audio_path = temp_dir / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ], check=True, capture_output=True)

    print("[2/7] Separating vocals from background (Demucs)...")
    demucs_out = temp_dir / "demucs_out"
    subprocess.run([
        sys.executable, "-m", "demucs", "--two-stems=vocals",
        str(audio_path), "-o", str(demucs_out)
    ], check=True)
    stem_dir = demucs_out / "htdemucs" / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    background_path = stem_dir / "no_vocals.wav"

    print("[3/7] Transcribing...")
    segments = transcribe(vocals_path, duration=duration)
    print(f"    {len(segments)} segments")

    print("[4/7] Diarizing...")
    speaker_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers)
    clean_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers, exclusive=True)
    tagged_segments = assign_speakers(segments, speaker_spans)

    speakers = sorted(set(s.speaker for s in clean_spans))
    print(f"    speakers: {speakers}")
    ref_dir = temp_dir / "speaker_references"
    ref_dir.mkdir(exist_ok=True)
    speaker_references = {}
    for speaker in speakers:
        ref_path = ref_dir / f"{speaker}.wav"
        extract_speaker_audio(vocals_path, clean_spans, speaker, ref_path, max_duration=20.0)
        speaker_references[speaker] = ref_path
    fallback_reference = next(iter(speaker_references.values())) if speaker_references else None

    print("[5/7] Loading Chatterbox (once, reused for every fit attempt)...")
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from components import chatterbox_patch

    chatterbox_patch.apply()
    device = best_device()
    print(f"    device: {device}")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    synth_dir = temp_dir / "synthesis"
    synth_dir.mkdir(exist_ok=True)
    attempt_counter = {"n": 0}

    def make_synthesize_fn(reference_audio: Path):
        def synthesize_fn(hindi_text: str) -> tuple[str, float]:
            attempt_counter["n"] += 1
            out_path = synth_dir / f"attempt_{attempt_counter['n']:04d}.wav"
            wav = model.generate(hindi_text, language_id=target_lang, audio_prompt_path=str(reference_audio))
            ta.save(str(out_path), wav, model.sr)
            _, trimmed_dur = trim_trailing_silence(str(out_path), str(out_path))
            return str(out_path), trimmed_dur
        return synthesize_fn

    print("[6/7] Fitting + synthesizing each segment...")
    synthesized = []
    before_after = []  # (pre-fit first-attempt duration, final duration, target)
    # Subtitle timing is anchored to the ORIGINAL English slot (seg.start/
    # seg.end), not wherever the audio actually lands in the final mix --
    # that's the whole point: if the subtitle (fixed to the original cue)
    # and the Hindi voice you hear don't start together, that's visible
    # desync you can catch by eye, not just in a metrics table.
    srt_entries = []
    for i, seg in enumerate(tagged_segments):
        text = seg.text.strip()
        if not text:
            continue

        target_dur = seg.end - seg.start
        reference_audio = speaker_references.get(seg.speaker, fallback_reference)
        synthesize_fn = make_synthesize_fn(reference_audio)

        result = fit_segment(
            english_text=text,
            target_duration=target_dur,
            translate_fn=lambda t: translate_one(t, target_lang),
            synthesize_fn=synthesize_fn,
        )
        best = result.best

        final_path = synth_dir / f"seg_{i:04d}.wav"
        final_dur = fit_duration(best.audio_path, str(final_path), target_dur)

        synthesized.append(SynthesizedSegment(
            start=seg.start, end=seg.end, duration=final_dur,
            path=str(final_path), speaker=seg.speaker, text=best.hindi_text,
        ))
        before_after.append((result.attempts[0].duration, final_dur, target_dur))

        deviation = final_dur - target_dur
        srt_entries.append((
            seg.start, seg.end,
            f"{best.hindi_text}\n[target {target_dur:.2f}s | actual {final_dur:.2f}s | Δ{deviation:+.2f}s]"
        ))

        tries = len(result.attempts)
        tag = "1st try" if result.fit_on_first_try else f"{tries} tries"
        print(f"  seg {i+1}/{len(tagged_segments)} [{seg.speaker}] target={target_dur:.2f}s "
              f"first_attempt={result.attempts[0].duration:.2f}s final={final_dur:.2f}s ({tag})  "
              f"\"{best.hindi_text[:40]}\"")

    print("[7/7] Assembling final video...")
    output_video = Path("output/local_fit_test.mp4")
    output_video.parent.mkdir(exist_ok=True)
    assemble(
        synthesized_segments=synthesized,
        background_path=background_path,
        input_video_path=video_path,
        output_video_path=output_video,
        temp_dir=temp_dir / "assembly",
    )

    print("Burning sync-verification subtitles (Hindi text + timing readout, "
          "timed to the ORIGINAL English slot)...")
    srt_path = temp_dir / "sync_check.srt"
    write_srt(srt_entries, srt_path)
    subtitled_video = Path("output/local_fit_test_subtitled.mp4")
    burn_subtitles(output_video, srt_path, subtitled_video)

    print("\n" + "=" * 60)
    print("BEFORE (first translation attempt, no fitting) vs AFTER (fit + speed-adjust):")
    before_segs = [Segment(start=0, end=t, text="x") for _, _, t in before_after]
    before_synth = [SynthesizedSegment(start=0, end=t, duration=d, path="", speaker=None, text="")
                     for d, _, t in before_after]
    after_synth = [SynthesizedSegment(start=0, end=t, duration=d, path="", speaker=None, text="")
                    for _, d, t in before_after]
    print("  BEFORE:", duration_match_metrics(before_segs, before_synth))
    print("  AFTER: ", duration_match_metrics(before_segs, after_synth))
    print(f"\nFinal video (no subtitles) -> {output_video}")
    print(f"Final video (sync-check subtitles) -> {subtitled_video}")

    print("\nSpeaker-similarity vs. each speaker's (now short, curated) reference clip:")
    embedder = load_speaker_embedder(HF_TOKEN)
    ref_embeds = {sp: embedder(str(path)) for sp, path in speaker_references.items()}
    import numpy as np
    sims_by_speaker: dict[str, list[float]] = {}
    for seg in synthesized:
        if seg.speaker not in ref_embeds:
            continue
        emb = embedder(seg.path)
        ref = ref_embeds[seg.speaker]
        sim = float(np.dot(ref, emb) / (np.linalg.norm(ref) * np.linalg.norm(emb)))
        sims_by_speaker.setdefault(seg.speaker, []).append(sim)
    for sp, sims in sims_by_speaker.items():
        print(f"  {sp}: n={len(sims)} avg={sum(sims)/len(sims):.3f} min={min(sims):.3f} max={max(sims):.3f}")


if __name__ == "__main__":
    main()
