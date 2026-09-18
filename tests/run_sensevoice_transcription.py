"""
Transcribes audio with SenseVoiceSmall -- replaces Whisper as of this
session's testing (see components/transcription_sensevoice.py for the full
before/after evidence: Whisper fell into a 13x "Yeah." repetition loop on a
real 5-minute video, costing 8.36s of real drift; SenseVoiceSmall produced
zero such artifacts on the identical audio while running ~8-9x faster).

Must run in the isolated .venv_sensevoice environment: funasr pulls its own
transformers/tokenizers versions, and this project has been burned before
by installing a new model's dependencies into the shared main env (that's
exactly why Chatterbox and BandIt each get their own venv -- see
components/synthesis_chatterbox.py and components/cass_separation.py).

SenseVoice's native output is one block of text per file with WORD-level
timestamps (not Whisper-style sentence segments), so this reconstructs
sentence-level {start, end, text} segments by splitting on sentence-ending
punctuation and using the enclosed words' own timestamps -- matching the
Segment contract the rest of the pipeline expects.

Usage:
    .venv_sensevoice/bin/python3 tests/run_sensevoice_transcription.py <audio_path> <output_json_path>
"""
import json
import re
import sys
from pathlib import Path


def normalize_split_numbers(text: str) -> str:
    """
    SenseVoice's own final `text` field correctly merges spoken digit
    sequences into real numbers (e.g. "2024"), but the per-word `words`
    array used below for timestamps exposes the pre-merge, digit-by-digit
    tokenization ("2", "0", "2", "4") -- so rebuilding sentence text by
    joining `words` (needed to get per-word timing) reintroduces exactly
    the digit-splitting SenseVoice had already fixed. Confirmed directly:
    the same audio's `text` field says "2024" while its `words` array is
    ['2','0','2','4'] for that span. This re-merges what the join broke.
    """
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'(?<=\d) (?=\d)', '', text)
    text = re.sub(r'(?<=\d) , (?=\d)', ',', text)
    text = re.sub(r'(?<=\d) %', '%', text)
    text = re.sub(r'(?<=\d) ([xX])\b', r'\1', text)
    return text


def reconstruct_segments(text: str, timestamps: list, words: list) -> list[dict]:
    """
    SenseVoice returns one block of text (with inline <|lang|><|emotion|>...
    tags stripped by the caller before this function runs) plus a parallel
    list of per-word (start_ms, end_ms) timestamps. Groups words into
    sentence-level segments by splitting on ., ?, ! -- each segment's start/
    end comes from its first/last word's own timestamp, converted to seconds.
    """
    segments = []
    current_words = []
    current_ts = []

    for word, ts in zip(words, timestamps):
        current_words.append(word)
        current_ts.append(ts)
        if word.strip().endswith((".", "?", "!")):
            segments.append({
                "start": current_ts[0][0] / 1000.0,
                "end": current_ts[-1][1] / 1000.0,
                "text": normalize_split_numbers(" ".join(current_words)),
            })
            current_words, current_ts = [], []

    if current_words:
        segments.append({
            "start": current_ts[0][0] / 1000.0,
            "end": current_ts[-1][1] / 1000.0,
            "text": normalize_split_numbers(" ".join(current_words)),
        })

    return segments


def main():
    if len(sys.argv) < 3:
        print("Usage: run_sensevoice_transcription.py <audio_path> <output_json_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = Path(sys.argv[2])

    from funasr import AutoModel
    model = AutoModel(
        model="FunAudioLLM/SenseVoiceSmall",
        device="cpu",
        hub="hf",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
    )
    res = model.generate(
        input=audio_path,
        language="en",
        use_itn=True,
        output_timestamp=True,
        batch_size_s=60,
    )

    r = res[0]
    import re
    clean_text = re.sub(r"<\|.*?\|>", "", r["text"]).strip()
    segments = reconstruct_segments(clean_text, r["timestamp"], r["words"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(segments)} segments to {output_path}")


if __name__ == "__main__":
    main()
