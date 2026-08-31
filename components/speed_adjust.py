"""
Bounded, pitch-preserving tempo adjustment -- nudges a synthesized clip's
duration to match its target time slot as closely as possible. NOT the
primary duration-fit mechanism -- that's condensing and retranslating the
source text (see components/duration_fit.py). This only closes whatever
gap remains after picking the closest-fitting translation candidate.

Bidirectional (speed up OR slow down): near-exact timing means an
undershooting candidate needs stretching too, not just an overshooting one
needing compression. ffmpeg's atempo filter accepts ratios on either side
of 1.0 for exactly this, and -- unlike naive resampling, which changes
pitch along with speed -- keeps pitch constant either way.

Verifies the actual resulting duration after each pass (atempo's own
frame-boundary rounding means the requested ratio isn't always exactly
what comes out) and retries with a corrected ratio up to MAX_CORRECTIONS
times, rather than trusting one computed ratio blindly.
"""
import subprocess
from pathlib import Path

import soundfile as sf

MAX_RATIO = 1.3          # cap in EITHER direction -- beyond this, speech
MIN_RATIO = 1 / MAX_RATIO  # starts sounding unnaturally fast/slow
TOLERANCE_S = 0.05
MAX_CORRECTIONS = 2


def _duration(path: str) -> float:
    data, sr = sf.read(path)
    return len(data) / sr


def fit_duration(audio_path: str, out_path: str, target_dur: float) -> float:
    """
    Adjusts audio_path's tempo (sped up or slowed down) so its duration is
    as close to target_dur as the MAX_RATIO/MIN_RATIO safety bounds allow,
    writing the result to out_path. Returns the resulting duration.

    No-ops (just copies audio_path to out_path) if already within
    TOLERANCE_S of target_dur, or if target_dur <= 0.
    """
    current_dur = _duration(audio_path)

    if target_dur <= 0 or abs(current_dur - target_dur) <= TOLERANCE_S:
        if audio_path != out_path:
            data, sr = sf.read(audio_path)
            sf.write(out_path, data, sr)
        return current_dur

    source_path = audio_path
    for correction in range(MAX_CORRECTIONS):
        needed_ratio = current_dur / target_dur
        ratio = max(MIN_RATIO, min(needed_ratio, MAX_RATIO))
        at_cap = ratio == MIN_RATIO or ratio == MAX_RATIO

        # ffmpeg can't read and write the same file in one invocation --
        # always render to a scratch path, then move it into place.
        scratch = f"{out_path}.tmp{correction}.wav"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", source_path,
            "-filter:a", f"atempo={ratio:.5f}",
            scratch,
        ], check=True, capture_output=True)
        Path(scratch).replace(out_path)

        current_dur = _duration(out_path)
        source_path = out_path

        if abs(current_dur - target_dur) <= TOLERANCE_S or at_cap:
            break

    return current_dur
