"""
Central config for the modular pipeline components. Existing pipeline.py is
untouched and keeps its own inline constants — this is only used by the new
components/ modules as they're built.
"""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — avoids adding python-dotenv as a dependency
    just for one variable. Only sets keys not already in the environment,
    so a real `export HF_TOKEN=...` always takes precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(Path(__file__).resolve().parent / ".env")

# Read from the environment (or .env above), never hardcoded — see pyannote's
# gated model terms: https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN = os.environ.get("HF_TOKEN")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def best_device() -> str:
    """Picks the fastest available torch device: a CUDA GPU (rented/desktop
    Linux boxes), else Apple Silicon MPS (this Mac), else CPU. Checking CUDA
    first matters -- torch.backends.mps.is_available() is always False on
    non-Apple hardware, but code that only ever checked mps-or-cpu would
    silently run on CPU on a CUDA machine instead of using the GPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# XTTS-v2 language codes (must use these exact strings)
XTTS_LANG_MAP = {
    "en": "en", "es": "es", "fr": "fr", "de": "de",
    "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
    "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
    "zh": "zh-cn", "ja": "ja", "hu": "hu", "ko": "ko", "hi": "hi"
}
