#!/usr/bin/env bash
# Full pipeline, one command: raw video -> final dubbed video.
# Wraps dub_multispeaker.sh (which expects vocals/background tracks to
# already exist) with the audio extraction + separation steps it assumes
# were already done -- so a new video doesn't need those run by hand first.
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
DEMUCS_OUT="temp/demucs_out/${STEM}"
# Demucs names its output subfolder after the input file's own basename
# (without extension) -- AUDIO_PATH is "${STEM}_audio.wav", so this must match.
DEMUCS_VOCALS="${DEMUCS_OUT}/htdemucs/${STEM}_audio/vocals.wav"
DEMUCS_BACKGROUND="${DEMUCS_OUT}/htdemucs/${STEM}_audio/no_vocals.wav"
CASS_OUT="temp/cass_out/${STEM}"
VOCALS_PATH="${CASS_OUT}/speech.wav"
BACKGROUND_PATH="${CASS_OUT}/background.wav"

# See dub_multispeaker.sh for why this checks uname, not just -x: on this
# Mac, .venv_main is a stale rsync copy of the Linux venv and would pass an
# -x check while pointing at a broken interpreter.
if [ "$(uname)" = "Linux" ] && [ -x ./.venv_main/bin/python3 ]; then
    MAIN_PY="./.venv_main/bin/python3"
else
    MAIN_PY="${MAIN_PY:-/Users/chandreshpatel/.pyenv/versions/3.12.5/bin/python3}"
fi

# .venv_cass (BandIt separation, see components/cass_separation.py) is
# currently only built on this Mac -- no Linux/GPU-box build yet. Rather
# than hard-fail on any machine without it (a real Linux GPU box, Colab),
# fall back to Demucs's own vocals/background split directly below --
# BandIt only adds non-verbal-vocalization preservation on top of that,
# it's not required for the pipeline to run.
CASS_PY="${CASS_PY:-/Users/chandreshpatel/gpu-dubbing/.venv_cass/bin/python3}"
if [ ! -x "$CASS_PY" ] && [ -x ./.venv_cass/bin/python3 ]; then
    CASS_PY="./.venv_cass/bin/python3"
fi

mkdir -p temp output

if [ -f "$VOCALS_PATH" ] && [ -f "$BACKGROUND_PATH" ]; then
    echo "=== Speech/background already separated, skipping extraction + separation ==="
    echo "    (delete $VOCALS_PATH to force redoing this step)"
else
    echo "=== Extracting audio ==="
    ffmpeg -y -i "$VIDEO_PATH" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_PATH"

    echo "=== Stage 1: Demucs (dialogue vs. background music/effects) ==="
    # NOTE: --shifts=0 was tried here (forces Demucs's otherwise
    # unseeded/random test-time-augmentation shift to be deterministic --
    # confirmed via direct testing to make separation output byte-identical
    # across runs, at the cost of whatever quality Demucs's default shifts
    # value adds). Reverted at user's request pending further comparison of
    # actual output quality between the two settings -- see conversation
    # for the reproducibility tradeoff this reintroduces.
    $MAIN_PY -m demucs --two-stems=vocals -n htdemucs "$AUDIO_PATH" -o "$DEMUCS_OUT"

    if [ -x "$CASS_PY" ]; then
        echo "=== Stage 2: BandIt on Demucs's vocals (pure speech vs. non-verbal vocalizaams get merged into background instead of silently lost/mistranslated; seecomponents/nonverbal_separation.py; CPU-only, ~45s/6s-chunk of the vocals track) ==="
        $CASS_PY tests/run_nonverbal_separation.py "$DEMUCS_VOCALS" "$DEMUCS_BACKGROUND" "$VOCALS_PATH" "$BACKGROUND_PATH"
    else
        echo "=== .venv_cass not found on this machine -- skipping BandIt, using Demucs's vocals/background split directly ==="
        mkdir -p "$CASS_OUT"
        cp "$DEMUCS_VOCALS" "$VOCALS_PATH"
        cp "$DEMUCS_BACKGROUND" "$BACKGROUND_PATH"
    fi
fi

echo "=== Running prepare/synthesize/assemble ==="
./dub_multispeaker.sh "$VIDEO_PATH" "$VOCALS_PATH" "$BACKGROUND_PATH" "$OUTPUT_PATH" "$TARGET_LANG" $NUM_SPEAKERS
