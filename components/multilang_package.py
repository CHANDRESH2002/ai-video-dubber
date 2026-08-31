"""
Packages one video + N language tracks (audio, and optionally subtitles)
into a single MKV file with each track tagged by language and a human-
readable title, so a viewer can pick their language from their player's own
audio/subtitle menu (VLC, Plex, Kodi, smart TVs, most desktop/mobile media
apps) -- no custom player needed. MKV specifically, not MP4: MKV's own
subtitle-as-a-stream support (plain SRT muxes in directly) is more broadly
handled by media players than MP4's equivalent (mov_text), and switching
between multiple embedded audio tracks is exactly what the format was
built for (it's the same mechanism DVDs/Blu-rays use).

This is deliberately just a muxing step -- it doesn't know or care how each
audio/subtitle track was produced (original recording, IndicTrans2 +
Chatterbox, condensed via Ollama, whatever). Feed it finished per-language
files and a language list.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

# ffmpeg/Matroska expect ISO 639-2 (3-letter) language codes for metadata,
# not the ISO 639-1 (2-letter) codes the rest of this codebase uses
# (config.py's XTTS_LANG_MAP, target_lang args, etc.) -- translate at the
# boundary here rather than growing a second code system everywhere else.
ISO_639_2 = {
    "en": "eng", "hi": "hin", "fr": "fra", "de": "deu", "es": "spa",
    "it": "ita", "pt": "por", "pl": "pol", "tr": "tur", "ru": "rus",
    "nl": "nld", "cs": "ces", "ar": "ara", "zh": "zho", "ja": "jpn",
    "hu": "hun", "ko": "kor",
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "pl": "Polish",
    "tr": "Turkish", "ru": "Russian", "nl": "Dutch", "cs": "Czech",
    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "hu": "Hungarian",
    "ko": "Korean",
}


@dataclass
class LanguageTrack:
    lang_code: str          # ISO 639-1, e.g. "hi" -- must be a key in ISO_639_2
    audio_path: Path
    subtitle_path: Path | None = None   # None = no subtitle track for this language
    is_default: bool = False            # which track plays without the viewer choosing


def mux_multilang(
    video_path: Path,
    tracks: list[LanguageTrack],
    output_path: Path,
) -> None:
    """
    video_path: source video -- only its video stream is used (re-encoding
        audio/subtitles doesn't require touching this, so it's stream-copied).
    tracks: one entry per selectable language, in the order they should
        appear in the player's menu. Exactly one should have is_default=True
        (falls back to the first track if none do).
    """
    if not tracks:
        raise ValueError("mux_multilang needs at least one LanguageTrack")
    for t in tracks:
        if t.lang_code not in ISO_639_2:
            raise ValueError(f"no ISO 639-2 code known for '{t.lang_code}' -- add it to ISO_639_2 above")

    if not any(t.is_default for t in tracks):
        tracks[0].is_default = True

    subtitle_tracks = [t for t in tracks if t.subtitle_path is not None]

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    for t in tracks:
        cmd += ["-i", str(t.audio_path)]
    for t in subtitle_tracks:
        cmd += ["-i", str(t.subtitle_path)]

    cmd += ["-map", "0:v:0"]
    for i in range(len(tracks)):
        cmd += ["-map", f"{1 + i}:a:0"]
    subtitle_input_offset = 1 + len(tracks)
    for i in range(len(subtitle_tracks)):
        cmd += ["-map", f"{subtitle_input_offset + i}:0"]

    cmd += ["-c:v", "copy", "-c:a", "aac", "-c:s", "srt"]

    for i, t in enumerate(tracks):
        iso3 = ISO_639_2[t.lang_code]
        name = LANGUAGE_NAMES.get(t.lang_code, t.lang_code)
        cmd += [f"-metadata:s:a:{i}", f"language={iso3}"]
        cmd += [f"-metadata:s:a:{i}", f"title={name}"]
        cmd += [f"-disposition:a:{i}", "default" if t.is_default else "0"]

    for i, t in enumerate(subtitle_tracks):
        iso3 = ISO_639_2[t.lang_code]
        name = LANGUAGE_NAMES.get(t.lang_code, t.lang_code)
        cmd += [f"-metadata:s:s:{i}", f"language={iso3}"]
        cmd += [f"-metadata:s:s:{i}", f"title={name}"]
        cmd += [f"-disposition:s:{i}", "default" if t.is_default else "0"]

    cmd += [str(output_path)]

    subprocess.run(cmd, check=True, capture_output=True)
    langs = ", ".join(LANGUAGE_NAMES.get(t.lang_code, t.lang_code) for t in tracks)
    print(f"Multi-language file → {output_path}  (audio: {langs}; "
          f"subtitles: {', '.join(LANGUAGE_NAMES.get(t.lang_code, t.lang_code) for t in subtitle_tracks) or 'none'})")
