# Dubbing pipeline — project context

Video dubbing pipeline: takes a source video (currently English), transcribes,
translates, and re-synthesizes speech in a target language (currently Hindi)
**cloning each original speaker's own voice**, then reassembles into a final
video with subtitles. Solo/unfunded founder project — prefer free/local
tooling over paid APIs where reasonable.

## The one command

```bash
./dub_from_video.sh <video_path> <output_path> <target_lang> [num_speakers]
```

There is no single `main.py` (despite earlier drafts of this doc claiming
one) — this shell script is the real one-command entry point. It handles
audio extraction + Demucs separation, then hands off to
`dub_multispeaker.sh`, which chains three Python stages across the two
environments (see Environments below) via a JSON manifest on disk:
`tests/dub_multispeaker_prepare.py` (main env) →
`tests/dub_multispeaker_synthesize.py` (`.venv_chatterbox`) →
`tests/dub_multispeaker_assemble.py` (main env). Read those three files plus
`components/*.py` for the whole pipeline flow. (`pipeline.py` at the repo
root is an older single-speaker/single-file baseline, superseded by this —
kept around for reference, not the thing to run.)

For a video whose vocals/background have already been separated (resuming
after a crash), the extraction+Demucs step auto-skips if
`temp/demucs_out/htdemucs/<stem>_audio/{vocals,no_vocals}.wav` already exist.

## Architecture

```
video.mp4
  -> ffmpeg: extract audio (16kHz mono wav)
  -> Demucs: separate vocals / background music
  -> faster-whisper (Linux) or mlx_whisper (Mac), task=translate -> English segments w/ timestamps
  -> pyannote.audio: diarize (who spoke when) -- run twice, overlap-aware + exclusive
  -> alignment: tag each transcript segment with a speaker
  -> IndicTrans2: translate English segments -> target language
  -> extract one clean reference clip per speaker (for voice cloning)
  == everything above: one process, main env ==
  -- write manifest.json, subprocess into .venv_chatterbox --
  -> Chatterbox Multilingual TTS: synthesize each segment, cloning that
     segment's speaker from their reference clip
  -> audio_trim.py: trim trailing dead air
  -> speed_adjust.py: bounded (<=1.3x) pitch-preserving speed-up if still over target
  -- subprocess returns, back in main env --
  -> assembly.py: place each speaker's segments on their own timeline
     (a segment can't start before the previous one from the SAME speaker
     finished -- this is the mechanism behind "drift", see below),
     mix with background audio, mux into video
  -> burn source-language subtitles
  -> final dubbed video
```

## Environments -- why there are two, and it's not optional

`components/synthesis_chatterbox.py` (Chatterbox TTS) **cannot** be imported
into the same Python process as diarization/translation/evaluation. Proved
repeatedly, not a config issue:
- `pyannote-audio` requires `torch>=2.8.0`
- `chatterbox-tts` pins its own `transformers` version incompatible with the
  main stack's `transformers==4.57.6`
- `unbabel-comet` (CometKiwi, quality scoring) requires
  `torchmetrics<0.11.0`; `pyannote-audio` requires `torchmetrics>=1.6.1` --
  literally no version satisfies both

So: **main env** (`requirements.txt`) has pyannote, IndicTrans2, Demucs,
faster-whisper/mlx-whisper. **`.venv_chatterbox`** has only `chatterbox-tts`
(which pulls its own pinned torch/transformers). They talk via a JSON
manifest file on disk + a subprocess call, never a shared import.

`requirements-eval.txt` (just `unbabel-comet`) is separate again -- it
conflicts with pyannote too, so evaluation/quality-scoring scripts need
their own third venv if you ever run them.

## Setting up on a fresh machine (e.g. a rented GPU box)

```bash
python3 -m venv .venv_main
source .venv_main/bin/activate
pip install -r requirements.txt   # torch/torchaudio/pyannote/torchcodec are version-pinned -- don't loosen without re-checking why (see the CUDA-compat gotcha below; there's no git history in this repo to check instead)
deactivate

python3 -m venv .venv_chatterbox
.venv_chatterbox/bin/pip install chatterbox-tts
.venv_chatterbox/bin/pip install "setuptools<81"   # chatterbox-tts's watermarking dep (resemble-perth) imports pkg_resources, which newer setuptools removed -- fails as a confusing TypeError deep in from_pretrained(), not an ImportError
```

**Gotchas hit setting up on a real rented GPU box (E2E Networks/TIR
platform, but some are general):**
- A container-level `PIP_CONSTRAINT` env var forced an unrelated non-PyPI
  torch build -- `unset PIP_CONSTRAINT` before any pip install if you hit a
  torch dependency-resolution error.
- Unpinned `torch` grabs whatever's newest, which can be compiled for a
  newer CUDA than the box's driver supports -- `torch.cuda.is_available()`
  silently returns `False` instead of erroring. Always verify with
  `python3 -c "import torch; print(torch.cuda.is_available())"` after any
  torch install on a new box. Current main-env pin (`torch==2.8.0`+cu126,
  installed via `--index-url https://download.pytorch.org/whl/cu126`) was
  chosen specifically to satisfy pyannote's `>=2.8.0` floor while staying
  CUDA-backward-compatible; re-derive if the pin ever needs to change.
- `config.best_device()` checks CUDA before MPS before CPU -- don't
  reintroduce a bare MPS-or-CPU check anywhere, it silently runs on CPU on
  any CUDA machine.
- `Path.write_text()`/`read_text()` without `encoding="utf-8"` uses the
  locale default, which is ASCII on a minimal container (no `LANG` set) --
  crashes on Hindi/Devanagari text. Always pass `encoding="utf-8"`
  explicitly for anything that touches translated text.
- yt-dlp: the Ubuntu apt package is often too outdated for current YouTube
  extraction. Get the standalone binary instead:
  `curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp && chmod +x /usr/local/bin/yt-dlp`

**Run things resiliently on a remote box**: always start long jobs inside
`tmux` (`tmux new -s dub`, detach with `Ctrl+B` `D`, reattach with
`tmux attach -t dub`) -- an SSH disconnect kills a bare foreground process,
but a `tmux` session keeps running on the box regardless of your local
connection, laptop sleep, etc. `caffeinate -i -w <PID>` is the Mac-side
equivalent for local long-running jobs.

## Chatterbox bugs found and fixed (`components/chatterbox_patch.py`)

Found by directly instrumenting `AlignmentStreamAnalyzer`
(`chatterbox/models/t3/inference/alignment_stream_analyzer.py` in the
installed package -- these are real upstream bugs, patched via monkeypatch
at runtime, not edits to the installed package):

1. **Rambling / trailing silence / run-to-run duration variance.** Root
   cause: the analyzer knows internally when text is "done"
   (`self.complete`), but the model's own stop-token probability stays near
   zero regardless, and the built-in safety nets (`long_tail`,
   `alignment_repetition`, `token_repetition`) often don't fire for
   short/simple text -- generation just rambles on for up to 150+ extra
   steps until a repetition randomly triggers a stop. Fix: force EOS
   `COMPLETE_GRACE_STEPS` (20) steps after `self.complete` first fires,
   instead of waiting on those safety nets.
2. **Hook leak.** Every `generate()` call registers new forward hooks via
   `_add_attention_spy()` without ever removing the old ones -- harmless in
   a fresh-subprocess-per-call setup, but in a long-lived process (e.g. a
   persistent worker) hooks accumulate forever and every stale one still
   fires on every forward pass. Fix: capture handles, clear the previous
   call's hooks at the start of each new `generate()`.
3. **Crash on very short text** (S<=5 text tokens, e.g. a one-word Hindi
   utterance): `A[self.completed_at:, :-5].max(dim=1)` slices to zero
   columns when S<=5 and throws `IndexError`. The EOS-suppression check
   right above already guards this exact case (`S > 5`); this one didn't.
   Hit at segment 335/491 of a real 32-minute video run. Fixed with the
   same guard.

Because fix #3 is inside `step()`'s body, the whole method is reimplemented
in the patch (not wrapped) -- keep it in sync with upstream if
`chatterbox-tts` is ever upgraded.

## Duration-fit strategy (why lines run long, and what fights it)

Chatterbox's own generation-time fix above handles the worst of it. On top:
- `audio_trim.py`: trims trailing dead air.
- `speed_adjust.py`: bounded (max 1.3x) pitch-preserving speed-up for
  whatever's still over target after the above. Runs even on
  already-synthesized/resumed segments (cheap, ffmpeg-only, no GPU).
- **Not yet re-integrated into `main.py`**: the condense-and-retranslate
  fallback loop (shorten the English source via an LLM and retranslate if
  a line still doesn't fit) -- this existed as a working scratchpad
  prototype earlier in the project but needs a clean redesign before
  production use, since it needs to cross between the two venvs repeatedly
  (translation needs main env, synthesis needs Chatterbox env) and the
  original prototype did this via a slow, fragile per-call subprocess
  reload. Worth revisiting once other priorities are settled.

**"Overflow" vs "drift"** (see any dub sync report): overflow is a single
segment's own duration vs. its target, in isolation. Drift is how far a
segment's *real* placement in the final assembly has been pushed later than
its original timestamp, because `assembly.py`'s per-speaker cursor
(`start = max(seg.start, cursor)`) means an overflowing segment delays
every later segment from the *same speaker* -- this compounds across a long
video. A segment can have near-zero overflow but huge drift, inherited from
everything before it.

## Validated results so far

- 25-segment, ~1min test clip (`temp/conversation_sample.mp4`): full
  pipeline validated end-to-end via `main.py` on this Mac (MPS).
- 491-segment, ~32min real video (Alexandr Wang / YC interview,
  downloaded via yt-dlp): full run on a rented GPU box (NVIDIA L4, E2E
  Networks/TIR platform, CUDA confirmed working via `config.best_device()`).
  Before `speed_adjust` was wired in: +18.6% overall overflow, max drift
  87.1s. After wiring it in (cheap re-run, no GPU cost since segments were
  already synthesized): **+2.6% overflow, max drift 33.7s**. A "Dub Sync
  Report" artifact (interactive HTML, per-segment English/Hindi timing +
  before/after drift chart) was built to visualize this -- see chat history
  or regenerate similarly from a fresh manifest + synthesized JSON pair if
  needed again.

## Known open issues

- **GPU box's CUDA was found disabled (as of 2026-08-29 audit)**: both
  venvs' `torch.cuda.is_available()` returned `False`, even though
  `nvidia-smi` shows the L4 healthy. Root cause: the container has
  `NVIDIA_VISIBLE_DEVICES=void` set, which stops the NVIDIA container
  runtime from wiring up real GPU access, despite `/dev/nvidia4` being
  bind-mounted (which is why `nvidia-smi`, going through NVML, still sees
  the card). This is a container/platform config issue on the E2E
  Networks/TIR side, not a code bug -- fix by relaunching the container
  with `NVIDIA_VISIBLE_DEVICES=all` (or the specific GPU's UUID), not by
  changing anything in this repo. Re-run
  `python3 -c "import torch; print(torch.cuda.is_available())"` after any
  relaunch to confirm before starting a real job.
- **`.venv_main`/`.venv_chatterbox` can't run from the Mac side.** They're
  the real Linux venvs (Linux `.so` binaries; `bin/python3` symlinks
  resolve to macOS's own system Python when read from this side) made
  visible here only because `/Users/chandreshpatel/gpu-dubbing` is an NFS
  mount into the same directory the GPU box sees as `/root/dubbing` (see
  "Remote GPU box" below) -- an `-x` executable check alone can't tell
  "visible" apart from "runnable." `dub_multispeaker.sh`/`dub_from_video.sh`
  now check `uname` before trusting them, falling back to the pyenv
  interpreter (which has both stacks installed natively for the Mac) on
  non-Linux.

- **First-segment transcription timing**: on the YC video, the transcribed
  first segment starts at exactly `0.0s` with real, correct text -- but
  this may be a Whisper-family quirk (the very first segment has no prior
  context to calibrate against, so its start time is less reliable than
  later segments) rather than the true speech onset, meaning the Hindi dub
  may start audibly before/with the video while the real English speech
  starts a beat later. Not yet confirmed with a real by-ear timestamp check
  against the source video -- do that first before changing any code.
- Hindi -> English (reverse direction) was validated for translation
  quality only (IndicTrans2 `indictrans2-indic-en-dist-200M`, avg CometKiwi
  0.854 on 25 real pairs, even better than the forward direction's 0.803)
  -- Chatterbox synthesizing English output and the condense-loop's Hindi
  paraphrase quality are both untested.
- Project structure cleanup discussed but only partly done: `tests/`
  contains actual pipeline stages (`dub_multispeaker_prepare/synthesize/
  assemble.py`), not tests -- a rename to something like `pipelines/` was
  proposed but not executed (would need re-syncing + re-verifying the GPU
  box afterward). `scratchpad_reports/` (the Dub Sync Report generator) is
  informally where generated reports live; not yet formalized.

## Remote GPU box (current, may be torn down after use)

- `ssh root@101.53.140.27`, then `tmux attach -t dub` to reattach to any
  running session.
- `/root/dubbing` on the box **is** this repo -- this Mac's
  `/Users/chandreshpatel/gpu-dubbing` (where this file lives) is an NFS
  mount (`fuse-t:/GPU-Dubbing`, confirmed via `mount`) straight into that
  same directory, not a separate local copy. Any edit made through either
  path lands on the same disk instantly -- no rsync needed, and the
  earlier version of this doc claiming otherwise was wrong. This is also
  why `.venv_main`/`.venv_chatterbox` are visible from the Mac at all: it's
  not that they were copied here, they're the real Linux venvs, just
  NFS-visible. They still can't be *executed* from the Mac side (wrong
  binary format) -- see the `uname` check in `dub_multispeaker.sh` /
  `dub_from_video.sh`.
- Actually running anything (Python, `nvidia-smi`, `tmux`) still requires
  SSHing in -- the NFS mount only shares the filesystem, not execution.
- `.venv_main` and `.venv_chatterbox` are already set up there per the
  gotchas above.
