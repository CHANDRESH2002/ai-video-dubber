#!/usr/bin/env bash
# Runs the full dubbing pipeline + evaluation report on the Bear monologue
# test clip. Wraps tests/test_evaluation.py so there's one file to run
# instead of retyping the full python command each time.
#
# Usage:
#   ./dub.sh [target_lang] [output_path]
#
# Examples:
#   ./dub.sh                                  # Hindi -> output/bear_dubbed_hi.mp4
#   ./dub.sh hi output/bear_dubbed_hi_v2.mp4   # Hindi -> custom output path
#   ./dub.sh es output/bear_dubbed_es.mp4      # Spanish -> custom output path

set -euo pipefail
cd "$(dirname "$0")"

TARGET_LANG="${1:-hi}"
OUTPUT="${2:-output/bear_dubbed_${TARGET_LANG}.mp4}"

COQUI_TOS_AGREED=1 /Users/chandreshpatel/.pyenv/versions/3.12.5/bin/python3 tests/test_evaluation.py \
  temp/bear_full.mp4 \
  temp/demucs_out/htdemucs/audio/vocals.wav \
  temp/demucs_out/htdemucs/audio/no_vocals.wav \
  "$OUTPUT" \
  "$TARGET_LANG" \
  1
