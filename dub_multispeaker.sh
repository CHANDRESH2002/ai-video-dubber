#!/usr/bin/env bash
# Multi-speaker Chatterbox dub pipeline. Chains 3 stages across 2 Python
# environments (see components/synthesis_chatterbox.py for why Chatterbox
# needs its own isolated venv):
#   1. prepare   (main env)        -- diarize, transcribe, align, translate
#   2. synthesize (.venv_chatterbox) -- per-speaker Chatterbox voice cloning
#   3. assemble  (main env)        -- mix, mux, burn subtitles
#
# Usage:
#   ./dub_multispeaker.sh <video_path> <vocals_path> <background_path> <output_path> <target_lang> [num_speakers]

set -euo pipefail
cd "$(dirname "$0")"

VIDEO_PATH="$1"
VOCALS_PATH="$2"
BACKGROUND_PATH="$3"
OUTPUT_PATH="$4"
TARGET_LANG="${5:-hi}"
NUM_SPEAKERS="${6:-}"

# Prefer the project-local venvs on Linux (rented GPU box) -- that's the
# only platform they're actually built for. On this Mac, `.venv_main` and
# `.venv_chatterbox` are stale rsync copies of the Linux venvs (Linux .so
# binaries; their bin/python3 symlink even resolves to macOS's system
# Python), so an `-x` check alone would silently "succeed" into a broken
# interpreter -- checking `uname` first avoids that. Mac always uses the
# pyenv interpreter, which has both stacks installed directly.
if [ "$(uname)" = "Linux" ] && [ -x ./.venv_main/bin/python3 ]; then
    MAIN_PY="./.venv_main/bin/python3"
else
    MAIN_PY="${MAIN_PY:-/Users/chandreshpatel/.pyenv/versions/3.12.5/bin/python3}"
fi
if [ "$(uname)" = "Linux" ] && [ -x ./.venv_chatterbox/bin/python3 ]; then
    CHATTERBOX_PY="./.venv_chatterbox/bin/python3"
else
    CHATTERBOX_PY="${CHATTERBOX_PY:-/Users/chandreshpatel/.pyenv/versions/3.12.5/bin/python3}"
fi

echo "=== Stage 1/3: prepare (main env) ==="
$MAIN_PY tests/dub_multispeaker_prepare.py "$VIDEO_PATH" "$VOCALS_PATH" "$TARGET_LANG" $NUM_SPEAKERS

echo "=== Stage 2/3: synthesize (chatterbox env) ==="
$CHATTERBOX_PY tests/dub_multispeaker_synthesize.py

echo "=== Stage 3/3: assemble (main env) ==="
$MAIN_PY tests/dub_multispeaker_assemble.py "$BACKGROUND_PATH" "$OUTPUT_PATH"

echo "Done → $OUTPUT_PATH"
