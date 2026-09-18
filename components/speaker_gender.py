"""
Speaker gender classification -- audeering/wav2vec2-large-robust-24-ft-age-gender,
used to fix Hindi's first-person grammatical gender agreement (see
components/gender_agreement_hindi.py). IndicTrans2 has no way to know the
speaker's gender from text alone (English "I" carries none), so this
recovers the missing signal directly from the speaker's own voice --
validated on 12 real speaker clips (mostly >95% confident) plus two full
real speeches, where correcting on this signal took a female speaker's
first-person lines from 1/9 correct to 9/9, with zero regressions on an
already-correct male speaker.

Runs in the MAIN env -- transformers/torch are already installed there for
IndicTrans2, this needs nothing extra.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

_MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"

_processor = None
_model = None


class _ModelHead(nn.Module):
    """Classification head."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Age/gender classifier on top of Wav2Vec2-Large-Robust."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = _ModelHead(config, 1)
        self.gender = _ModelHead(config, 3)  # [female, male, child]
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def _load_model():
    global _processor, _model
    if _model is not None:
        return
    _processor = Wav2Vec2Processor.from_pretrained(_MODEL_NAME)
    _model = _AgeGenderModel.from_pretrained(_MODEL_NAME)
    _model.eval()


def _load_as_16k_mono(path: Path) -> np.ndarray:
    """Loads ANY audio/video file as 16kHz mono float32 via ffmpeg --
    sidesteps soundfile's limited format support (no m4a/mp3/mp4 etc)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "converted.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(path), "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", str(wav_path),
        ], check=True, capture_output=True)
        signal, sr = sf.read(str(wav_path))
        assert sr == 16000
        return signal.astype(np.float32)


def classify_speaker_gender(audio_path: Path, confidence_threshold: float = 0.7) -> str:
    """
    Returns "male", "female", or "unknown" if neither probability clears
    confidence_threshold -- callers should skip gender-agreement correction
    entirely on "unknown" rather than guess (e.g. one real speaker in
    testing came back only 68% female / 30% child, a genuinely weak read).
    """
    _load_model()
    signal = _load_as_16k_mono(audio_path)

    inputs = _processor(signal, sampling_rate=16000)
    x = torch.from_numpy(inputs["input_values"][0]).reshape(1, -1)

    with torch.no_grad():
        _, _age_logits, gender_logits = _model(x)

    female, male, _child = gender_logits[0].tolist()
    if female >= confidence_threshold:
        return "female"
    if male >= confidence_threshold:
        return "male"
    return "unknown"
