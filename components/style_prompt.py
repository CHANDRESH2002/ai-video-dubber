"""
Auto-generates a natural-language style prompt per segment, from the
ORIGINAL ENGLISH audio -- the real performance (whisper, anger, pace)
lives in how the source speaker actually said it, independent of which
language gets synthesized. Feeds components/synthesis_voxcpm.py's
parenthetical style-instruction mechanism.

Two steps, mirroring how TextrolSpeech itself was built (see
scratchpad_reports/textrolspeech_samples.html for real examples):
  1. Measure discrete labels off the waveform + get an emotion tag from
     SenseVoice's built-in SER (separate from whichever model is doing the
     actual transcription text -- SenseVoice vs Qwen3-ASR is an unrelated,
     unresolved choice from other experimentation in this project).
  2. Turn those labels into one short natural sentence via Ollama, same
     request pattern as components/translation.py's translate_dubbing_style().

EXPERIMENTAL: pitch/speed/volume bucket thresholds below are first-pass
heuristics, not empirically tuned -- expect to revisit after looking at
real output on a real clip.
"""
import subprocess
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SENSEVOICE_PY = ROOT / "sensevoice-experiment" / ".venv_sensevoice" / "bin" / "python3"

_OLLAMA_URL = "http://localhost:11434/api/chat"
_OLLAMA_MODEL = "qwen2.5:14b"


def _extract_clip(audio_path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data[int(start * sr):int(end * sr)], sr


def measure_prosody(audio_path: Path, start: float, end: float, text: str) -> dict:
    """Pitch/speed/volume, each bucketed low/normal/high -- same three
    factors TextrolSpeech itself uses (minus gender, which we already get
    from components/speaker_gender.py)."""
    import librosa

    clip, sr = _extract_clip(audio_path, start, end)
    duration = max(end - start, 0.01)

    rms = float(np.sqrt(np.mean(clip ** 2))) if len(clip) else 0.0
    volume = "low" if rms < 0.02 else "high" if rms > 0.08 else "normal"

    word_count = len(text.split())
    words_per_sec = word_count / duration if word_count else 0.0
    speed = "low" if words_per_sec < 2.0 else "high" if words_per_sec > 3.3 else "normal"

    try:
        f0, voiced_flag, _ = librosa.pyin(
            clip, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        median_pitch = float(np.median(voiced_f0)) if len(voiced_f0) else 0.0
    except Exception:
        median_pitch = 0.0
    pitch = "low" if 0 < median_pitch < 140 else "high" if median_pitch > 220 else "normal"

    return {"pitch": pitch, "speed": speed, "volume": volume}


def classify_emotion(audio_path: Path, start: float, end: float) -> str:
    """Emotion tag via SenseVoice's SER, run as a subprocess in its own
    venv (same isolation pattern as every other cross-venv call in this
    project). Falls back to "neutral" on any failure -- this is a nice-to-
    have signal, not worth failing the whole prepare stage over."""
    clip, sr = _extract_clip(audio_path, start, end)
    if len(clip) == 0:
        return "neutral"

    tmp_path = Path("temp/_style_prompt_clip.wav")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path), clip, sr)

    try:
        result = subprocess.run(
            [str(SENSEVOICE_PY), str(ROOT / "tests" / "run_sensevoice_emotion.py"), str(tmp_path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return result.stdout.strip() or "neutral"
    except Exception as e:
        print(f"  [style_prompt] emotion classification failed, defaulting to neutral: {e}")
        return "neutral"


def generate_style_prompt(labels: dict) -> str:
    """One Ollama call, turning discrete labels into a short comma-separated
    style FRAGMENT -- matching VoxCPM2's own documented convention for
    controllable cloning (its technical report's example: "speaking very
    fast, bright and full"), not a full grammatical sentence. Confirmed via
    the model's own source (voxcpm/core.py, voxcpm2.py) that this prefix is
    tokenized as ordinary text with no separate channel -- its length
    directly inflates the generation-length cap, so verbose sentence-style
    output (the previous version of this prompt) was a real contributor to
    the duration-overflow bug traced in scratchpad_reports/voxcpm2_dataflow.html."""
    prompt = (
        f"Given these voice attributes: pitch={labels['pitch']}, speaking speed={labels['speed']}, "
        f"volume={labels['volume']}, emotion={labels['emotion']} -- write a SHORT style-control "
        f"phrase, 3-6 words, comma-separated fragments, NOT a full sentence (no subject/verb "
        f"grammar). Example of the exact style wanted: \"speaking very fast, bright and full\". "
        f"Another example: \"low pitch, slow and hesitant, quiet\". "
        f"Respond with ONLY the phrase, no quotes, no preamble."
    )
    try:
        response = requests.post(_OLLAMA_URL, json={
            "model": _OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }, timeout=30)
        response.raise_for_status()
        return response.json()["message"]["content"].strip().strip('"').strip("'").strip()
    except Exception as e:
        print(f"  [style_prompt] Ollama generation failed, falling back to raw labels: {e}")
        return f"{labels['pitch']} pitch, {labels['speed']} speed, {labels['volume']} volume, {labels['emotion']} tone"


def get_style_prompt(vocals_path: Path, start: float, end: float, text: str) -> str:
    """Single entry point for the prepare stage."""
    labels = measure_prosody(vocals_path, start, end, text)
    labels["emotion"] = classify_emotion(vocals_path, start, end)
    return generate_style_prompt(labels)
