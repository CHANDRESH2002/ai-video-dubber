"""
Stage 3/3 of the multi-speaker Chatterbox dub pipeline. Runs in the MAIN
environment. Reads the manifest written by stage 2 (synthesis, chatterbox
env), assembles the final video, and burns in English subtitles.

Usage:
    python tests/dub_multispeaker_assemble.py <background_path> <output_path>
"""

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_types import SynthesizedSegment
from components.assembly import assemble
from components.subtitles import write_srt, burn_subtitles


def main():
    if len(sys.argv) < 3:
        print("Usage: python tests/dub_multispeaker_assemble.py <background_path> <output_path>")
        sys.exit(1)

    background_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    prep_manifest = json.loads(Path("temp/multispeaker_manifest.json").read_text())
    synth_manifest = json.loads(Path("temp/multispeaker_synthesized.json").read_text())

    synthesized = [SynthesizedSegment(**s) for s in synth_manifest["synthesized"]]
    video_path = Path(synth_manifest["video_path"])

    print("Assembling...")
    unsubtitled_path = output_path.with_name(output_path.stem + "_nosub" + output_path.suffix)
    assemble(
        synthesized_segments=synthesized,
        background_path=background_path,
        input_video_path=video_path,
        output_video_path=unsubtitled_path,
        temp_dir=Path("temp/multispeaker_assembly"),
    )

    print("Burning subtitles (original-language source text)...")
    entries = [
        (s["start"], s["end"], src)
        for s, src in zip(prep_manifest["segments"], prep_manifest["source_texts"])
        if src.strip()
    ]
    srt_path = Path("temp/multispeaker.srt")
    write_srt(entries, srt_path)
    burn_subtitles(unsubtitled_path, srt_path, output_path)

    print(f"Final video → {output_path}")


if __name__ == "__main__":
    main()
