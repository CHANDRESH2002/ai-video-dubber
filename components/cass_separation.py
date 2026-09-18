"""
Cinematic Audio Source Separation (CASS) via BandIt -- replaces Demucs's
vocals/no_vocals split. Demucs is a music-source-separation model with no
concept of "dialogue vs. everything else"; it routes non-verbal
vocalizations (laughs, cries, screams) inconsistently, and since
assembly.py discards the vocals stem's content after transcription, a
non-verbal sound Demucs happened to call "vocals" was silently lost from
the final dub. BandIt splits into speech / music / effects / residual
instead, trained on a narrower LibriSpeech-style definition of "speech" --
narrower in a way that happens to work in our favor: it routes almost any
non-neutral vocal performance (not just clean dialogue) OUT of "speech"
and into music/effects, which -- because we then reuse music+effects+residual
as the background track -- means it survives in the final mix instead of
disappearing. Measured on 4 deliberately hard real clips (scream+music,
crying-while-singing, reaction video, laugh-track): 98-100% of that
moment's energy preserved into the final mix, vs. Demucs's 21-99% (badly
leaky specifically on the singing-cry and laugh-track cases, since Demucs
correctly recognizes those AS vocals and routes them to the stem we throw
away).

speech.wav plays the role today's vocals.wav did (diarization,
transcription, speaker reference extraction). music+effects+residual.wav
(not music+effects.wav alone) plays the role of no_vocals.wav -- the
residual term matters: residual = mixture - (speech+music+effects), so
speech + (music+effects+residual) reconstructs the original mixture
exactly, with zero energy loss by construction. Demucs's own vocals +
no_vocals split has no such guarantee (measured summing to LESS than the
original on a real clip).

IMPORTANT: this module can only run inside the isolated `.venv_cass`
environment -- BandIt's `environment.yaml` pins python=3.10, torch==2.0.0,
incompatible with the main environment's newer torch (used by
IndicTrans2/pyannote). Run this file's logic via `.venv_cass/bin/python3`,
never the main interpreter. Currently only built on this Mac (Python
3.10.18 via pyenv); would need rebuilding to run on the Linux GPU box.

Known cost: CPU-only here (no GPU, and BandIt's own inference.py has no
MPS-awareness even on Apple Silicon), roughly 45s per 6-second audio chunk.
A 40s clip takes ~5 minutes; a 30-minute video would take hours. Worth
revisiting GPU access before running this on long videos.

Setup (already done on this Mac, documented here for a fresh machine):
    pyenv install 3.10.18
    ~/.pyenv/versions/3.10.18/bin/python3 -m venv .venv_cass
    .venv_cass/bin/pip install torch==2.0.0 torchaudio==2.0.0 torchvision==0.15.0
    .venv_cass/bin/pip install "numpy==1.26.4"   # torch 2.0.0's ABI expects numpy 1.x
    .venv_cass/bin/pip install lightning torchmetrics pandas fire librosa spafe \\
        openunmix pedalboard torch-audiomentations asteroid tensorboard tensorboardx
    # the line above WILL silently upgrade torch again via transitive deps -- re-pin:
    .venv_cass/bin/pip install torch==2.0.0 torchaudio==2.0.0 torchvision==0.15.0 \\
        --force-reinstall --no-deps
    .venv_cass/bin/pip install "numpy==1.26.4"
    .venv_cass/bin/pip install "pytorch-lightning==2.0.9" --no-deps --force-reinstall
    .venv_cass/bin/pip install "setuptools<81"   # same pkg_resources issue as chatterbox
    git clone https://github.com/kwatcharasupat/bandit.git cass-separation-experiment/bandit
    # ERB-48 checkpoint (375MB) from Zenodo record 10160698:
    curl -L -o cass-separation-experiment/models/dnr-3s-erb48-l1snr/checkpoints/dnr-3s-erb48-l1snr.ckpt \\
        "https://zenodo.org/records/10160698/files/dnr-3s-erb48-l1snr.ckpt?download=1"
    cp cass-separation-experiment/bandit/expt/dnr-3s-erb48-l1snr.yaml \\
        cass-separation-experiment/models/dnr-3s-erb48-l1snr/hparams.yaml
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANDIT_DIR = ROOT / "cass-separation-experiment" / "bandit"
MODEL_DIR = ROOT / "cass-separation-experiment" / "models" / "dnr-3s-erb48-l1snr"
CKPT_PATH = MODEL_DIR / "checkpoints" / "dnr-3s-erb48-l1snr.ckpt"
MODEL_NAME = "dnr-3s-erb48-l1snr"


def separate(audio_path: Path, speech_out_path: Path, background_out_path: Path, work_dir: Path) -> None:
    """
    Runs BandIt on audio_path, then copies its speech.wav to
    speech_out_path and its music+effects+residual.wav (the exact,
    zero-loss "everything except speech" reconstruction) to
    background_out_path.
    """
    # Resolve to absolute paths before building the subprocess command --
    # BandIt's inference.py runs with cwd=BANDIT_DIR, so a relative path
    # here would be looked up inside the bandit repo, not the caller's cwd.
    audio_path = audio_path.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PROJECT_ROOT"] = str(BANDIT_DIR)

    print(f"  [cass_separation] running BandIt (ERB-48) on {audio_path}...")
    subprocess.run(
        [
            sys.executable, "inference.py", "inference",
            f"--ckpt_path={CKPT_PATH}",
            f"--file_path={audio_path}",
            f"--model_name={MODEL_NAME}",
            f"--output_dir={work_dir}",
        ],
        cwd=str(BANDIT_DIR),
        env=env,
        check=True,
    )

    speech_out_path.parent.mkdir(parents=True, exist_ok=True)
    background_out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_find_output(work_dir, "speech.wav"), speech_out_path)
    shutil.copy(_find_output(work_dir, "music+effects+residual.wav"), background_out_path)
    print(f"  [cass_separation] wrote {speech_out_path} and {background_out_path}")


def _find_output(work_dir: Path, filename: str) -> Path:
    """BandIt sometimes writes directly into work_dir and sometimes nests
    output one level deeper under a track-name subfolder, depending on
    internal batching (predict_step's include_track_name = batch_size > 1
    in bandit/core/__init__.py) -- not something worth depending on exactly,
    so search for it instead of assuming a fixed path."""
    direct = work_dir / filename
    if direct.exists():
        return direct
    matches = list(work_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"BandIt did not produce {filename} anywhere under {work_dir}")
    return matches[0]
