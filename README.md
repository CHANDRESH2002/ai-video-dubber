# Dubbing Pipeline

Takes a source video, transcribes it, translates it, and re-synthesizes speech
in one or more target languages **cloning each original speaker's own voice**,
fits each line's timing to match the original speaker's cadence, and can
package multiple languages as selectable audio/subtitle tracks in one file
(like a DVD/Netflix language menu) instead of one file per language.

This started as a solo/unfunded exploration and is still early — expect rough
edges. See [Known issues](./CLAUDE.md#known-open-issues) before relying on it
for anything real. Contributions and issue reports welcome.

## Try it now

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CHANDRESH2002/ai-video-dubber/blob/main/AI_Video_Dubber_Demo.ipynb)

Upload your own short video and get it dubbed into Hindi -- cloned speaker
voices, timing-matched, subtitles burned in -- using Google Colab's free GPU.
No local setup needed. Click the badge above.

## Quick start

```bash
cp .env.example .env   # fill in your own HF_TOKEN
./dub_from_video.sh <video_path> <output_path> <target_lang> [num_speakers]
```

See **[CLAUDE.md](./CLAUDE.md)** for the full pipeline architecture, environment
setup (two separate venvs are required — this is not optional, see why in
there), known issues, and validated results. That file is the actual source of
truth for how this project works; keep it updated as the pipeline changes.

## Layout

- `components/` — pipeline stages (diarization, transcription, translation,
  synthesis, assembly, evaluation), each independently importable.
  - `duration_fit.py` — retries a line's translation (via a local LLM
    condensing the English source) when the synthesized audio doesn't fit
    its original time slot, picking whichever attempt lands closest.
  - `speed_adjust.py` — bounded, pitch-preserving tempo correction (either
    direction) for whatever timing gap remains after duration-fit.
  - `multilang_package.py` — muxes one video + N language audio/subtitle
    tracks into a single MKV with proper language tags, playable in
    VLC/Plex/Kodi/smart TVs via their normal audio/subtitle menu.
- `tests/` — despite the name, `dub_multispeaker_*.py` here are the real
  pipeline-stage entry points (see CLAUDE.md's Known Issues); `test_*.py` are
  manual validation scripts requiring a real `HF_TOKEN` and GPU, not an
  automated suite. `local_duration_fit_run.py` and `local_multilang_run.py`
  are single-process (no GPU-box venv split needed) runners for iterating on
  the duration-fit and multi-language features on a Mac.
- `pipeline.py` — older single-speaker/single-file baseline, superseded by the
  `components/` pipeline. Kept for reference.
- `dub_from_video.sh` / `dub_multispeaker.sh` / `dub.sh` — orchestration.

## Requirements

Not committed here (large, platform-specific): `input/`, `output/`, `temp/`,
and both virtual environments. Follow the setup steps in CLAUDE.md to build
`.venv_main` and `.venv_chatterbox` on a fresh machine.
