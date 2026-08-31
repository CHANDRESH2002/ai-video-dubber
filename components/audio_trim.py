"""
Trims trailing dead-air off a synthesized clip. Chatterbox sometimes tacks
on 1-5+ seconds of near-silence after the real speech content ends (see
components/chatterbox_patch.py for the generation-side fix to the worst of
this) -- this catches whatever's left by finding the last window whose
energy is above a fraction of the clip's peak, plus a small buffer so words
aren't clipped short.
"""
import numpy as np
import soundfile as sf


def trim_trailing_silence(
    audio_path: str, out_path: str,
    silence_ratio: float = 0.20, buffer_s: float = 0.15, window_s: float = 0.05,
) -> tuple[float, float]:
    """Returns (original_duration, trimmed_duration)."""
    data, sr = sf.read(audio_path)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    original_dur = len(mono) / sr

    window = max(int(window_s * sr), 1)
    energies = []
    for i in range(0, len(mono), window):
        chunk = mono[i:i + window]
        rms = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0.0
        energies.append(rms)

    peak = max(energies) if energies else 0.0
    if peak == 0:
        sf.write(out_path, data, sr)
        return original_dur, original_dur

    threshold = peak * silence_ratio
    last_loud_window = 0
    for idx, e in enumerate(energies):
        if e > threshold:
            last_loud_window = idx

    cut_sample = min(len(mono), (last_loud_window + 1) * window + int(buffer_s * sr))
    trimmed = data[:cut_sample]
    sf.write(out_path, trimmed, sr)
    trimmed_dur = len(trimmed) / sr
    return original_dur, trimmed_dur
