"""
Second-stage separation: pulls non-verbal vocalizations (laughs, cries,
screams) OUT of Demucs's vocals.wav before it ever reaches transcription.

Why two stages instead of BandIt alone (see components/cass_separation.py
for BandIt's original standalone role): BandIt run directly on a full,
noisy mixture (crosstalk, continuous background noise/music) catastrophically
misclassifies ordinary dialogue as "music" -- confirmed on a real
multi-speaker action-set clip, ~92% of windows lost. But Demucs's own
vocals/no_vocals split is excellent at the first cut (dialogue vs.
background music, 93-99% correct on every real clip tested) and only
struggles with one narrower thing: telling real speech apart from
non-verbal vocalizations that got bundled into the same "vocals" stem.

That narrower problem is exactly BandIt's strength, and by running it on
Demucs's already background-free vocals.wav instead of the raw mixture, it
never has to fight background noise/music at the same time -- confirmed on
a real clip: BandIt-on-vocals correctly transcribes real dialogue at a
laugh's boundary ("Repulsive even. Yeah.") that a fixed-window audio-event
tagger corrupted into "Impulsive even.", while still fully suppressing the
laugh itself from the speech stem. BandIt's mask-based separation gives
smooth, partial energy transitions at boundaries (not a hard binary cut),
which is what avoids the tagger's word-clipping failure mode.

speech.wav (BandIt's output on vocals.wav) replaces vocals.wav as the input
to transcription. music+effects+residual.wav (BandIt's "not speech" output)
gets summed with Demucs's own no_vocals.wav into one merged background
track, so a non-verbal vocalization plays through in the final mix exactly
like any other background sound, instead of being transcribed/translated/
replaced by TTS.

Must run inside `.venv_cass` -- same constraint as cass_separation.py.
"""
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from components.cass_separation import separate as bandit_separate


def sum_audio(path_a: Path, path_b: Path, out_path: Path) -> None:
    """Sums two audio files sample-for-sample (padding the shorter to match),
    so both tracks' content survives in the final mix. Written as plain
    addition rather than an ffmpeg amix filter to avoid amix's automatic
    loudness renormalization, which would quietly change the level of
    content the rest of the pipeline hasn't touched."""
    sig_a, sr_a = sf.read(path_a)
    sig_b, sr_b = sf.read(path_b)
    if sr_a != sr_b:
        raise ValueError(f"sample rate mismatch: {path_a}={sr_a} vs {path_b}={sr_b}")
    if sig_a.ndim > 1:
        sig_a = sig_a.mean(axis=1)
    if sig_b.ndim > 1:
        sig_b = sig_b.mean(axis=1)

    n = max(len(sig_a), len(sig_b))
    sig_a = np.pad(sig_a, (0, n - len(sig_a)))
    sig_b = np.pad(sig_b, (0, n - len(sig_b)))
    mixed = sig_a + sig_b

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mixed, sr_a)


def remove_nonverbal(demucs_vocals_path: Path, demucs_background_path: Path,
                      speech_out_path: Path, background_out_path: Path, work_dir: Path) -> None:
    """
    Runs BandIt on Demucs's vocals.wav to split it into pure speech vs.
    non-verbal vocalizations, then merges the non-verbal content into
    Demucs's own background track -- so background_out_path ends up with
    everything that isn't spoken dialogue (music/effects + laughs/cries),
    and speech_out_path has only what should actually be transcribed.
    """
    work_dir = work_dir.resolve()
    nonverbal_out = work_dir / "nonverbal_from_vocals.wav"
    bandit_speech_out = work_dir / "bandit_speech.wav"

    print(f"  [nonverbal_separation] running BandIt on Demucs vocals: {demucs_vocals_path}")
    bandit_separate(demucs_vocals_path, bandit_speech_out, nonverbal_out, work_dir / "bandit_work")

    speech_out_path.parent.mkdir(parents=True, exist_ok=True)
    background_out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(bandit_speech_out, speech_out_path)

    print(f"  [nonverbal_separation] merging non-verbal content into background track")
    sum_audio(demucs_background_path, nonverbal_out, background_out_path)
    print(f"  [nonverbal_separation] wrote {speech_out_path} and {background_out_path}")
