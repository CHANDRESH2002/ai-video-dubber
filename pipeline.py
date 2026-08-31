"""
Baseline Dubbing Pipeline
=========================
Input  : video file in SOURCE_LANG
Output : dubbed video in TARGET_LANG, preserving the original speaker's voice

Stages:
  1. Extract audio from video
  2. Separate voice from background music/SFX  (Demucs)
  3. Transcribe voice to text                  (Whisper via MLX)
  4. Translate text                            (Google Translate)
  5. Synthesize dubbed voice                   (XTTS-v2)
  6. Assemble final video                      (ffmpeg)
"""

import os
import sys
import json
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_VIDEO  = "input/test_clip.mp4"
SOURCE_LANG  = "hi"    # language of the original video  (hi=Hindi, es=Spanish, fr=French ...)
TARGET_LANG  = "en"    # language you want to dub into   (en=English, es=Spanish ...)
OUTPUT_VIDEO = "output/dubbed.mp4"

# XTTS-v2 language codes (must use these exact strings)
XTTS_LANG_MAP = {
    "en": "en", "es": "es", "fr": "fr", "de": "de",
    "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
    "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
    "zh": "zh-cn", "ja": "ja", "hu": "hu", "ko": "ko", "hi": "hi"
}

TEMP = Path("temp")
TEMP.mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)


# ── STAGE 1 : EXTRACT AUDIO ──────────────────────────────────────────────────

def extract_audio():
    """Pull the audio track out of the video as a mono 16kHz WAV."""
    out = TEMP / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", INPUT_VIDEO,
        "-vn",                      # no video
        "-acodec", "pcm_s16le",    # uncompressed PCM
        "-ar", "16000",             # 16 kHz (Whisper's native rate)
        "-ac", "1",                 # mono
        str(out)
    ], check=True, capture_output=True)
    print(f"[1/6] Audio extracted  →  {out}")
    return out


# ── STAGE 2 : SEPARATE VOICE FROM BACKGROUND ─────────────────────────────────

def separate_voice(audio_path: Path):
    """
    Demucs splits audio into:
      vocals.wav    – the dialogue / singing
      no_vocals.wav – music + sound effects
    We keep no_vocals.wav to mix back in at the end.
    """
    demucs_out = TEMP / "demucs_out"
    subprocess.run([
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",       # only split into vocals / no_vocals
        str(audio_path),
        "-o", str(demucs_out)
    ], check=True)

    # Demucs puts output under:  demucs_out/<model>/<input_stem>/vocals.wav
    # Default model is htdemucs
    stem_name   = audio_path.stem                         # "audio"
    base        = demucs_out / "htdemucs" / stem_name
    vocals      = base / "vocals.wav"
    background  = base / "no_vocals.wav"

    print(f"[2/6] Voice separated  →  vocals: {vocals.name}  |  bg: {background.name}")
    return vocals, background


# ── STAGE 3 : TRANSCRIBE ─────────────────────────────────────────────────────

def transcribe(vocals_path: Path):
    """
    MLX-Whisper runs entirely on the M4's Neural Engine — very fast.

    Uses Whisper's built-in task="translate" instead of task="transcribe":
    Whisper decodes straight from audio to English text in one pass, rather
    than us transcribing in SOURCE_LANG first and translating the text after.
    That matters for code-switched speech (e.g. Hindi speakers dropping in
    English words/numbers) — transcribing forces every word into a single
    language's script, so embedded English gets phonetically mangled into
    Devanagari ("which is" → "विच इज") before translation ever sees it.
    Decoding straight to English sidesteps that step entirely.

    Trade-off: task="translate" only ever outputs English, regardless of
    TARGET_LANG. If TARGET_LANG isn't English, translate() below uses this
    clean English as a pivot rather than translating from the messier
    source-language transcript.

    Returns a list of segments: [{id, text, start, end}, ...]
    """
    import mlx_whisper

    print(f"[3/6] Transcribing with Whisper large-v3 (MLX)...")
    result = mlx_whisper.transcribe(
        str(vocals_path),
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        task="translate",
        language=SOURCE_LANG,
        word_timestamps=False,      # segment-level timestamps are enough
        verbose=False
    )

    segments = result["segments"]

    # Whisper can hallucinate extra segments past the end of real audio
    # (e.g. "Closed Captioning provided by ..."), and they reliably land
    # right at the true content boundary — sometimes even a hair before it,
    # since audio extraction/separation can pad the file by a few ms. Using
    # the source video's real duration (not the possibly-padded vocals.wav)
    # with a safety buffer catches these without cutting real trailing speech.
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        INPUT_VIDEO
    ], capture_output=True, text=True, check=True)
    video_duration = float(probe.stdout.strip())
    segments = [seg for seg in segments if seg["start"] < video_duration - 0.5]

    # Save so you can inspect / edit without re-running
    seg_path = TEMP / "segments.json"
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"[3/6] Transcribed  →  {len(segments)} segments  →  {seg_path}")
    return segments


# ── STAGE 4 : TRANSLATE ──────────────────────────────────────────────────────

def translate(segments: list):
    """
    segments[i]["text"] is already English (Whisper translated it directly
    in transcribe()). If TARGET_LANG is English, just use it as-is. Otherwise
    translate that clean English pivot into TARGET_LANG via deep-translator —
    translating from English is far more reliable than translating from the
    original code-switched source language would have been.
    """
    print(f"[4/6] Preparing {TARGET_LANG} text...")

    if TARGET_LANG == "en":
        for seg in segments:
            seg["translated"] = seg["text"].strip()
    else:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="en", target=TARGET_LANG)
        for seg in segments:
            original = seg["text"].strip()
            seg["translated"] = translator.translate(original) if original else ""

    trans_path = TEMP / "translated.json"
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"[4/6] Text ready  →  {trans_path}")
    return segments


# ── STAGE 5 : SYNTHESIZE DUBBED VOICE ────────────────────────────────────────

def synthesize(segments: list, reference_audio: Path):
    """
    XTTS-v2:  zero-shot voice cloning.
    Give it 6+ seconds of the speaker's voice and it will say anything
    in that same voice, in the target language.

    reference_audio = the isolated vocals file (from Demucs) — this is the
    speaker sample the model clones from.

    Duration matching: XTTS has no "make this exactly N seconds" control, so
    a naive single synthesis pass routinely runs longer than the original
    segment's time slot (translated English is often more syllables than the
    source, and XTTS's default pace doesn't match the speaker's). We close
    that gap in two steps:
      1. Synthesize once at natural pace to measure how long the text takes,
         then re-synthesize at XTTS's own `speed` parameter tuned toward the
         target duration — this changes the generated prosody itself, so it
         sounds like the voice naturally talking faster/slower rather than
         audio played back at the wrong speed.
      2. Whatever gap remains gets closed with a capped, pitch-preserving
         ffmpeg time-stretch (atempo). The cap matters: forcing a 4x-too-long
         segment to fit would need extreme stretching that turns speech into
         unintelligible mush, so we bound how much either step can push and
         accept residual overflow rather than destroy intelligibility.
    """
    from TTS.api import TTS
    from config import best_device

    device = best_device()
    print(f"[5/6] Loading XTTS-v2 on device: {device} (first run downloads ~2 GB)...")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    xtts_lang  = XTTS_LANG_MAP.get(TARGET_LANG, "en")
    seg_dir    = TEMP / "segments_audio"
    seg_dir.mkdir(exist_ok=True)

    MIN_SPEED   = 0.7   # XTTS speed multiplier bounds — beyond these the
    MAX_SPEED   = 1.7   # generated prosody starts sounding unnatural
    MAX_STRETCH = 1.3   # additional ffmpeg atempo compression allowed on top

    audio_segments = []

    for i, seg in enumerate(segments):
        text = seg.get("translated", "").strip()
        if not text:
            continue

        target_dur = seg["end"] - seg["start"]
        out_path = seg_dir / f"seg_{i:04d}.wav"

        # Pass 1: natural pace, just to measure how long this text takes
        tts.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio),
            language=xtts_lang,
            file_path=str(out_path)
        )
        natural_dur = sf.info(str(out_path)).duration

        # Pass 2: re-synthesize at an adjusted speed if natural pace misses target
        speed = natural_dur / target_dur if target_dur > 0 else 1.0
        speed = max(MIN_SPEED, min(MAX_SPEED, speed))
        if abs(speed - 1.0) > 0.05:
            tts.tts_to_file(
                text=text,
                speaker_wav=str(reference_audio),
                language=xtts_lang,
                file_path=str(out_path),
                speed=speed
            )

        actual_dur = sf.info(str(out_path)).duration

        # Close any remaining gap with a bounded, pitch-preserving time-stretch
        if actual_dur > target_dur * 1.05:
            stretch = min(actual_dur / target_dur, MAX_STRETCH)
            stretched_path = seg_dir / f"seg_{i:04d}_stretched.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", str(out_path),
                "-filter:a", f"atempo={stretch:.4f}",
                str(stretched_path)
            ], check=True, capture_output=True)
            out_path.unlink()
            stretched_path.rename(out_path)
            actual_dur = sf.info(str(out_path)).duration

        audio_segments.append({
            "path"     : str(out_path),
            "start"    : seg["start"],
            "end"      : seg["end"],
            "duration" : actual_dur,
            "text"     : text
        })

        overflow = actual_dur - target_dur
        flag = f"  ⚠ still {overflow:.1f}s over" if overflow > 0.15 else ""
        print(f"  segment {i+1}/{len(segments)}: [{seg['start']:.1f}s → {seg['end']:.1f}s] "
              f"target={target_dur:.1f}s actual={actual_dur:.1f}s{flag}  \"{text[:50]}\"")

    print(f"[5/6] Synthesized  →  {len(audio_segments)} audio segments")
    return audio_segments


# ── STAGE 6 : ASSEMBLE FINAL VIDEO ───────────────────────────────────────────

def assemble(audio_segments: list, background_path: Path):
    """
    1. Place each dubbed segment at its original timestamp on a silent canvas
       — unless a still-too-long earlier segment (see synthesize()'s duration
       matching) would make it overlap, in which case it's pushed to start
       right after the previous one instead. Summing overlapping speech
       waveforms produces unintelligible double-talk; a segment landing a
       bit later than its original cue is far less noticeable.
    2. Mix the dubbed speech with the background track.
    3. Merge final audio back into the original video (no video re-encode).
    """
    import subprocess

    # Get video duration via ffprobe
    probe = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        INPUT_VIDEO
    ], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())

    SR = 22050  # target sample rate for the assembled track (XTTS itself outputs 24kHz; resampled below)
    canvas = np.zeros(int(duration * SR) + SR)  # +1s padding

    cursor = 0  # next free sample position — keeps segments from overlapping
    for seg in audio_segments:
        data, seg_sr = sf.read(seg["path"])
        if data.ndim > 1:
            data = data.mean(axis=1)   # stereo → mono

        # Resample if needed (XTTS should always output 22050 but be safe)
        if seg_sr != SR:
            import torchaudio
            import torch
            t = torch.from_numpy(data).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(seg_sr, SR)
            data = resampler(t).squeeze(0).numpy()

        start = max(int(seg["start"] * SR), cursor)
        end   = start + len(data)

        if end > len(canvas):
            canvas = np.pad(canvas, (0, end - len(canvas)))

        canvas[start:end] += data
        cursor = end

    # Normalize to avoid clipping
    peak = np.max(np.abs(canvas))
    if peak > 0:
        canvas = canvas / peak * 0.85

    dubbed_speech = TEMP / "dubbed_speech.wav"
    sf.write(str(dubbed_speech), canvas, SR)

    # Mix dubbed speech + background
    final_audio = TEMP / "final_audio.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(dubbed_speech),
        "-i", str(background_path),
        "-filter_complex",
        # Speech at full volume, background at 80% so it doesn't overpower
        "[0:a]volume=1.0[speech]; [1:a]volume=0.8[bg]; [speech][bg]amix=inputs=2:duration=longest",
        str(final_audio)
    ], check=True, capture_output=True)

    # Merge audio into original video (copy video stream, no re-encode)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", INPUT_VIDEO,
        "-i", str(final_audio),
        "-c:v", "copy",         # keep original video exactly as-is
        "-c:a", "aac",
        "-map", "0:v:0",        # video from input 0
        "-map", "1:a:0",        # audio from input 1
        "-shortest",
        OUTPUT_VIDEO
    ], check=True)

    print(f"[6/6] Final video  →  {OUTPUT_VIDEO}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Baseline Dubbing Pipeline")
    print(f"  {SOURCE_LANG}  →  {TARGET_LANG}")
    print(f"  Input:  {INPUT_VIDEO}")
    print(f"  Output: {OUTPUT_VIDEO}")
    print("=" * 50 + "\n")

    audio_path              = extract_audio()
    vocals_path, bg_path    = separate_voice(audio_path)
    segments                = transcribe(vocals_path)
    segments                = translate(segments)
    audio_segments          = synthesize(segments, vocals_path)
    assemble(audio_segments, bg_path)

    print("\n" + "=" * 50)
    print("  Done. Open output/dubbed.mp4 to listen.")
    print("=" * 50)
