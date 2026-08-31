"""
Stage 2/3 of the multi-speaker Chatterbox dub pipeline. Must run in the
isolated .venv_chatterbox environment:

    .venv_chatterbox/bin/python3 tests/dub_multispeaker_synthesize.py

Reads the manifest written by dub_multispeaker_prepare.py (stage 1, main
env), synthesizes every segment with its own speaker's reference clip via
Chatterbox, and writes a second manifest for stage 3 (assembly, back in
the main env).
"""

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_types import Segment
from components.synthesis_chatterbox import synthesize


def main():
    manifest = json.loads(Path("temp/multispeaker_manifest.json").read_text())

    segments = [Segment(**s) for s in manifest["segments"]]
    speaker_references = {k: Path(v) for k, v in manifest["speaker_references"].items()}
    target_lang = manifest["target_lang"]

    output_dir = Path("temp/synthesis_multispeaker_chatterbox")
    synthesized = synthesize(segments, speaker_references, output_dir, target_lang=target_lang)

    out_path = Path("temp/multispeaker_synthesized.json")
    out_path.write_text(json.dumps({
        "video_path": manifest["video_path"],
        "synthesized": [dataclasses.asdict(s) for s in synthesized],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(synthesized)} synthesized segments)")


if __name__ == "__main__":
    main()
