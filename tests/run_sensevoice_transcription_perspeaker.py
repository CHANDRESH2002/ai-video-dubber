"""
Per-speaker variant of run_sensevoice_transcription.py: transcribes each
diarization-derived speaker block SEPARATELY, so a transcript segment can
never straddle a real speaker change (see components/alignment.py's
build_speaker_blocks() and the multi-speaker-attribution fix plan for why
the whole-file version made this structurally impossible to avoid --
SenseVoice splits sentences on punctuation, not on who's talking, so one
"sentence" could silently span a real speaker turn).

Loads the audio once, slices it per block, tags every resulting segment
directly with that block's already-known speaker -- no separate overlap-
based alignment step needed afterward.

Usage:
    .venv_sensevoice/bin/python3 run_sensevoice_transcription_perspeaker.py \
        <audio_path> <blocks_json_path> <output_json_path>

blocks_json: [{"start": float, "end": float, "speaker": str}, ...]
"""
import json
import sys
from pathlib import Path

from run_sensevoice_transcription import reconstruct_segments

MIN_BLOCK_DURATION = 0.3  # too short for reliable ASR -- same threshold
                          # used in run_qwen3asr_transcription.py


def main():
    if len(sys.argv) < 4:
        print("Usage: run_sensevoice_transcription_perspeaker.py <audio_path> <blocks_json_path> <output_json_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    blocks = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    output_path = Path(sys.argv[3])

    import re
    import soundfile as sf
    from funasr import AutoModel

    model = AutoModel(
        model="FunAudioLLM/SenseVoiceSmall",
        device="cpu",
        hub="hf",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        disable_pbar=True,
    )

    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    tmp_path = "/tmp/_sensevoice_perspeaker_block.wav"
    all_segments = []

    for block in blocks:
        start, end, speaker = block["start"], block["end"], block["speaker"]
        if end - start < MIN_BLOCK_DURATION:
            continue

        clip = data[int(start * sr):int(end * sr)]
        sf.write(tmp_path, clip, sr)

        res = model.generate(
            input=tmp_path,
            language="en",
            use_itn=True,
            output_timestamp=True,
            batch_size_s=60,
        )
        r = res[0]
        clean_text = re.sub(r"<\|.*?\|>", "", r["text"]).strip()
        if not clean_text:
            continue

        block_segments = reconstruct_segments(clean_text, r["timestamp"], r["words"])
        for seg in block_segments:
            # Segment timing is relative to this block's own slice --
            # offset back to the original clip's global timeline.
            all_segments.append({
                "start": start + seg["start"],
                "end": start + seg["end"],
                "text": seg["text"],
                "speaker": speaker,
            })
        print(f"  [{speaker}] {start:.2f}-{end:.2f}s: {len(block_segments)} segment(s)")

    all_segments.sort(key=lambda s: s["start"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_segments)} segments to {output_path}")


if __name__ == "__main__":
    main()
