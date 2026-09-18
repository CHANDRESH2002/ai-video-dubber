"""
Single-shot condense+retranslate worker, invoked as a subprocess from
components/synthesis_chatterbox.py -- which runs inside .venv_chatterbox
and can't import IndicTrans2 or call Ollama directly (see CLAUDE.md's
Environments section for why). One call = one retry attempt for one
overflowing segment: shrink the English source, retranslate it, hand the
new target-language text back to the caller to resynthesize and re-measure.

Must run under a main-env interpreter (has IndicTrans2 + reaches Ollama).

Usage:
    <main-env python> components/condense_retry_worker.py
    stdin:  {"text": str, "shrink_fraction": float, "target_lang": str, "gender": str | null}
    stdout: {"english": str, "translated": str}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.duration_fit import condense_english
from components.translation import translate_one
from components.number_localization import protect_numbers, restore_numbers
from components.gender_agreement_hindi import correct_gender_agreement


def main():
    req = json.loads(sys.stdin.read())
    text = req["text"]
    shrink_fraction = req["shrink_fraction"]
    target_lang = req["target_lang"]
    gender = req.get("gender")

    condensed = condense_english(text, shrink_fraction)

    if target_lang == "en":
        translated = condensed
    else:
        protected, placeholders = protect_numbers(condensed)
        translated = translate_one(protected, target_lang)
        translated = restore_numbers(translated, placeholders)
        if target_lang == "hi" and gender in ("male", "female"):
            translated = correct_gender_agreement(translated, gender)["text"]

    print(json.dumps({"english": condensed, "translated": translated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
