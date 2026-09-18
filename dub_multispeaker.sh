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
TTS_ENGINE="${TTS_ENGINE:-chatterbox}"   # chatterbox (default, validated) or voxcpm (new, see the integration plan)

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
# Unlike .venv_main/.venv_chatterbox, .venv_voxcpm is a REAL, working venv
# on this Mac too (built directly here, not an NFS-visible Linux copy) --
# voxcpm pulls its own torch (2.14.0), which would conflict with the main
# env's pinned torch if installed into the shared Mac pyenv interpreter,
# so it always gets its own venv on either platform. Only exists on this
# Mac so far -- Linux install is a follow-up, same situation .venv_cass is
# in today.
VOXCPM_PY="${VOXCPM_PY:-./.venv_voxcpm/bin/python3}"

# Exported so stage 2 (a different interpreter/venv) can shell back out to
# the main env for condense-and-retranslate retries on overflowing segments
# -- see components/synthesis_chatterbox.py and
# components/condense_retry_worker.py.
export MAIN_PY

echo "=== Stage 1/3: prepare (main env) ==="
$MAIN_PY tests/dub_multispeaker_prepare.py "$VIDEO_PATH" "$VOCALS_PATH" "$TARGET_LANG" $NUM_SPEAKERS

echo "=== Stage 2/3: synthesize ($TTS_ENGINE env) ==="
if [ "$TTS_ENGINE" = "voxcpm" ]; then
    $VOXCPM_PY tests/dub_multispeaker_synthesize_voxcpm.py
else
    $CHATTERBOX_PY tests/dub_multispeaker_synthesize.py
fi

echo "=== Stage 3/3: assemble (main env) ==="
$MAIN_PY tests/dub_multispeaker_assemble.py "$BACKGROUND_PATH" "$OUTPUT_PATH"

echo "Done → $OUTPUT_PATH"
