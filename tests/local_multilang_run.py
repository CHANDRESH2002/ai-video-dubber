"""
Local, single-process multi-language packaging test -- Mac only, same
reasoning as tests/local_duration_fit_run.py (this Mac's pyenv doesn't need
the venv split the Linux GPU box does).

Runs the shared pipeline stages ONCE (extract, separate, transcribe,
diarize, align, build speaker references), then for each target language
runs the full duration-fit + synthesize + speed-adjust + assemble flow, and
finally muxes the original English track plus every dubbed language into
one MKV file with selectable audio/subtitle tracks.

Usage:
    /Users/chandreshpatel/.pyenv/versions/3.12.5/bin/python3 \\
        tests/local_multilang_run.py <video_path> [num_speakers]

Target languages are set in TARGET_LANGS below.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HF_TOKEN, best_device
from data_types import SynthesizedSegment
from components.diarization import diarize, extract_speaker_audio
from components.transcription import transcribe
from components.alignment import assign_speakers
from components.translation import translate_one
from components.duration_fit import fit_segment
from components.speed_adjust import fit_duration
from components.audio_trim import trim_trailing_silence
from components.assembly import build_mixed_audio, compute_actual_placements
from components.subtitles import write_srt
from components.multilang_package import mux_multilang, LanguageTrack

TARGET_LANGS = ["hi", "fr"]   # dubbed languages, in addition to the original English track


def main():
    if len(sys.argv) < 2:
        print("Usage: local_multilang_run.py <video_path> [num_speakers]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    temp_dir = Path("temp/local_multilang_run")
    temp_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())

    print("[shared 1/5] Extracting audio...")
    audio_path = temp_dir / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ], check=True, capture_output=True)

    print("[shared 2/5] Separating vocals from background (Demucs)...")
    demucs_out = temp_dir / "demucs_out"
    subprocess.run([
        sys.executable, "-m", "demucs", "--two-stems=vocals",
        str(audio_path), "-o", str(demucs_out)
    ], check=True)
    stem_dir = demucs_out / "htdemucs" / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    background_path = stem_dir / "no_vocals.wav"

    print("[shared 3/5] Transcribing (English)...")
    english_segments = transcribe(vocals_path, duration=duration)
    print(f"    {len(english_segments)} segments")

    print("[shared 4/5] Diarizing + aligning...")
    speaker_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers)
    clean_spans = diarize(vocals_path, hf_token=HF_TOKEN, num_speakers=num_speakers, exclusive=True)
    tagged_segments = assign_speakers(english_segments, speaker_spans)

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

    print("[shared 5/5] Loading Chatterbox (once, reused across all languages)...")
    import torchaudio as ta
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from components import chatterbox_patch

    chatterbox_patch.apply()
    model = ChatterboxMultilingualTTS.from_pretrained(device=best_device())

    tracks: list[LanguageTrack] = []

    # --- Original English track: no synthesis needed, just the real audio
    # and a subtitle track from the plain English transcript. ---
    print("\n=== Building English (original) track ===")
    en_srt_path = temp_dir / "en.srt"
    write_srt(
        [(s.start, s.end, s.text) for s in tagged_segments if s.text.strip()],
        en_srt_path,
    )
    tracks.append(LanguageTrack(
        lang_code="en", audio_path=audio_path, subtitle_path=en_srt_path, is_default=True,
    ))

    # --- Each dubbed language: duration-fit -> synthesize -> speed-adjust
    # -> mix -> subtitle timed to the ACTUAL placement, not the original
    # English slot (this is a real subtitle track for viewers, not the
    # sync-verification one from local_duration_fit_run.py). ---
    for target_lang in TARGET_LANGS:
        print(f"\n=== Building {target_lang} track ===")
        synth_dir = temp_dir / f"synthesis_{target_lang}"
        synth_dir.mkdir(exist_ok=True)
        attempt_counter = {"n": 0}

        def make_synthesize_fn(reference_audio: Path):
            def synthesize_fn(text: str) -> tuple[str, float]:
                attempt_counter["n"] += 1
                out_path = synth_dir / f"attempt_{attempt_counter['n']:04d}.wav"
                wav = model.generate(text, language_id=target_lang, audio_prompt_path=str(reference_audio))
                ta.save(str(out_path), wav, model.sr)
                _, trimmed_dur = trim_trailing_silence(str(out_path), str(out_path))
                return str(out_path), trimmed_dur
            return synthesize_fn

        synthesized = []
        for i, seg in enumerate(tagged_segments):
            text = seg.text.strip()
            if not text:
                continue

            target_dur = seg.end - seg.start
            reference_audio = speaker_references.get(seg.speaker, fallback_reference)
            result = fit_segment(
                english_text=text,
                target_duration=target_dur,
                translate_fn=lambda t: translate_one(t, target_lang),
                synthesize_fn=make_synthesize_fn(reference_audio),
            )
            best = result.best

            final_path = synth_dir / f"seg_{i:04d}.wav"
            final_dur = fit_duration(best.audio_path, str(final_path), target_dur)

            synthesized.append(SynthesizedSegment(
                start=seg.start, end=seg.end, duration=final_dur,
                path=str(final_path), speaker=seg.speaker, text=best.hindi_text,
            ))
            print(f"  seg {i+1}/{len(tagged_segments)} [{seg.speaker}] target={target_dur:.2f}s "
                  f"final={final_dur:.2f}s  \"{best.hindi_text[:40]}\"")

        lang_audio = build_mixed_audio(
            synthesized, background_path, duration, temp_dir / f"mix_{target_lang}",
        )

        placements = compute_actual_placements(synthesized)
        srt_entries = [
            (*placements[id(seg)], seg.text)
            for seg in synthesized
        ]
        srt_path = temp_dir / f"{target_lang}.srt"
        write_srt(srt_entries, srt_path)

        tracks.append(LanguageTrack(lang_code=target_lang, audio_path=lang_audio, subtitle_path=srt_path))

    print("\n=== Muxing final multi-language file ===")
    output_path = Path("output/multilang_test.mkv")
    output_path.parent.mkdir(exist_ok=True)
    mux_multilang(video_path, tracks, output_path)


if __name__ == "__main__":
    main()
