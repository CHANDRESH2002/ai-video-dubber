"""
One-off recovery: rebuilds temp/multispeaker_synthesized.json from WAV files
that synthesis already wrote to disk successfully, for the case where
dub_multispeaker_synthesize.py's synthesis loop completed but crashed on
the final write_text() call (see git history for the UTF-8 encoding fix)
before the manifest itself got saved. Avoids re-running synthesis (slow,
and costs real money on a rented GPU) just to regenerate a JSON summary of
work that's already done.

Usage (main env, since it only needs soundfile, not chatterbox):
    python3 tests/dub_multispeaker_recover_manifest.py
"""
import json
from pathlib import Path

import soundfile as sf


def main():
    manifest = json.loads(Path("temp/multispeaker_manifest.json").read_text(encoding="utf-8"))
    segs = manifest["segments"]
    output_dir = Path("temp/synthesis_multispeaker_chatterbox")

    synthesized = []
    non_empty_idx = 0
    for seg in segs:
        if not seg["text"].strip():
            continue
        out_path = output_dir / f"seg_{non_empty_idx:04d}.wav"
        non_empty_idx += 1
        if not out_path.exists():
            print(f"MISSING: {out_path} -- this segment needs re-synthesis")
            continue
        data, sr = sf.read(str(out_path))
        duration = len(data) / sr
        synthesized.append({
            "start": seg["start"], "end": seg["end"], "duration": duration,
            "path": str(out_path), "speaker": seg["speaker"], "text": seg["text"],
        })

    out_path = Path("temp/multispeaker_synthesized.json")
    out_path.write_text(json.dumps({
        "video_path": manifest["video_path"],
        "synthesized": synthesized,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recovered {len(synthesized)}/{non_empty_idx} segments -> {out_path}")


if __name__ == "__main__":
    main()
