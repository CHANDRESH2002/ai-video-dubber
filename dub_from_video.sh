#!/usr/bin/env bash
# Full pipeline, one command: raw video -> final dubbed video.
# Wraps dub_multispeaker.sh (which expects vocals/background tracks to
# already exist) with the audio extraction + Demucs separation steps it
# assumes were already done -- so a new video doesn't need those run by
# hand first.
#
# Usage:
#   ./dub_from_video.sh <video_path> <output_path> <target_lang> [num_speakers]

set -euo pipefail
cd "$(dirname "$0")"

VIDEO_PATH="$1"
OUTPUT_PATH="$2"
TARGET_LANG="${3:-hi}"
NUM_SPEAKERS="${4:-}"

STEM="$(basename "$VIDEO_PATH")"
STEM="${STEM%.*}"
AUDIO_PATH="temp/${STEM}_audio.wav"
DEMUCS_OUT="temp/demucs_out"
VOCALS_PATH="${DEMUCS_OUT}/htdemucs/${STEM}_audio/vocals.wav"
BACKGROUND_PATH="${DEMUCS_OUT}/htdemucs/${STEM}_audio/no_vocals.wav"

# See dub_multispeaker.sh for why this checks uname, not just -x: on this
# Mac, .venv_main is a stale rsync copy of the Linux venv and would pass an
# -x check while pointing at a broken interpreter.
if [ "$(uname)" = "Linux" ] && [ -x ./.venv_main/bin/python3 ]; then
    MAIN_PY="./.venv_main/bin/python3"
else
    MAIN_PY="${MAIN_PY:-/Users/chandreshpatel/.pyenv/versions/3.12.5/bin/python3}"
fi

mkdir -p temp output

if [ -f "$VOCALS_PATH" ] && [ -f "$BACKGROUND_PATH" ]; then
    echo "=== Vocals/background already separated, skipping extraction + Demucs ==="
    echo "    (delete $VOCALS_PATH to force redoing this step)"
else
    echo "=== Extracting audio ==="
    ffmpeg -y -i "$VIDEO_PATH" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_PATH"

    echo "=== Separating vocals from background (Demucs) ==="
    $MAIN_PY -m demucs --two-stems=vocals "$AUDIO_PATH" -o "$DEMUCS_OUT"
fi

echo "=== Running prepare/synthesize/assemble ==="
./dub_multispeaker.sh "$VIDEO_PATH" "$VOCALS_PATH" "$BACKGROUND_PATH" "$OUTPUT_PATH" "$TARGET_LANG" $NUM_SPEAKERS
