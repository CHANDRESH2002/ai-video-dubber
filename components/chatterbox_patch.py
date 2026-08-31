"""
Fixes for three real Chatterbox bugs found while running this pipeline,
all in AlignmentStreamAnalyzer
(chatterbox/models/t3/inference/alignment_stream_analyzer.py):

1. Rambling / trailing silence / duration variance. Root cause: once the
   text is actually finished, the analyzer correctly detects this internally
   (self.complete flips True once the smoothed text position crosses S-3) --
   but nothing acts on that. The model's own raw stop-token probability
   stays near 0.0000 regardless, and the existing safety nets (long_tail,
   alignment_repetition, token_repetition) require conditions that often
   don't fire for short/simple text. So generation just keeps rambling
   until, by chance, the same token gets sampled twice in a row and
   token_repetition forces a stop -- anywhere from a few steps to 150+,
   which is what drove both the multi-second trailing silence/rambling and
   the run-to-run duration variance for identical input. Fix: once
   self.complete first becomes True, force EOS after COMPLETE_GRACE_STEPS
   more steps instead of waiting indefinitely on those safety nets.

2. Hook leak (only matters for a persistent/long-lived process, e.g. a
   worker that calls generate() many times without restarting). Every call
   to model.generate() creates a fresh AlignmentStreamAnalyzer, whose
   _add_attention_spy() calls target_layer.register_forward_hook(...) but
   discards the returned handle -- so the hook can never be removed. Hooks
   accumulate forever, and every stale hook still fires on every forward
   pass. Fix: capture the handle, and clear all handles from the previous
   call at the start of each new generate() call.

3. Crash on very short text (S <= 5 text tokens -- e.g. a one-word Hindi
   utterance). The original step() computes
   `A[self.completed_at:, :-5].max(dim=1)` unconditionally once
   self.complete is True; when S <= 5, `:-5` slices down to zero columns
   and .max() on an empty dimension raises
   `IndexError: max(): Expected reduction dim 1 to have non-zero size`.
   The EOS-suppression logic right above already guards the same S <= 5
   edge case (`if cur_text_posn < S - 3 and S > 5`) -- this one just
   forgot to. Fix: same S > 5 guard, defaulting alignment_repetition to
   False for tiny S (self.complete is already reliable there on its own).

Apply once per process via chatterbox_patch.apply(), before any
model.generate() call. Since fix #3 is inside the original step() body
(not before/after it), the whole method is reimplemented here rather than
wrapping the original -- keep this in sync with upstream if chatterbox-tts
is ever upgraded.
"""
import logging

import torch
from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

logger = logging.getLogger("chatterbox.models.t3.inference.alignment_stream_analyzer")

COMPLETE_GRACE_STEPS = 20

_original_step = AlignmentStreamAnalyzer.step


def _patched_step(self, logits, next_token=None):
    aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0)
    i, j = self.text_tokens_slice
    if self.curr_frame_pos == 0:
        A_chunk = aligned_attn[j:, i:j].clone().cpu()
    else:
        A_chunk = aligned_attn[:, i:j].clone().cpu()
    A_chunk[:, self.curr_frame_pos + 1:] = 0
    self.alignment = torch.cat((self.alignment, A_chunk), dim=0)
    A = self.alignment
    T, S = A.shape
    cur_text_posn = A_chunk[-1].argmax()
    discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
    if not discontinuity:
        self.text_position = cur_text_posn
    false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
    self.started = not false_start
    if self.started and self.started_at is None:
        self.started_at = T
    self.complete = self.complete or self.text_position >= S - 3
    if self.complete and self.completed_at is None:
        self.completed_at = T
    long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5)
    # Fix #3: guard S > 5 same as the EOS-suppression check below -- for
    # S <= 5 there aren't enough columns for A[:, :-5] to be non-empty.
    alignment_repetition = (
        self.complete and S > 5
        and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)
    )
    if next_token is not None:
        if isinstance(next_token, torch.Tensor):
            token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
        else:
            token_id = next_token
        self.generated_tokens.append(token_id)
        if len(self.generated_tokens) > 8:
            self.generated_tokens = self.generated_tokens[-8:]
    token_repetition = (
        len(self.generated_tokens) >= 3
        and len(set(self.generated_tokens[-2:])) == 1
    )
    if token_repetition:
        repeated_token = self.generated_tokens[-1]
        logger.warning(f"🚨 Detected 2x repetition of token {repeated_token}")
    # Suppress EoS to prevent early termination
    if cur_text_posn < S - 3 and S > 5:  # Only suppress if text is longer than 5 tokens
        logits[..., self.eos_idx] = -2**15
    if long_tail or alignment_repetition or token_repetition:
        logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
        logits = -(2**15) * torch.ones_like(logits)
        logits[..., self.eos_idx] = 2**15
    self.curr_frame_pos += 1

    # Fix #1: force EOS shortly after self.complete fires, instead of
    # waiting indefinitely on long_tail/alignment_repetition/token_repetition.
    if self.complete and self.completed_at is not None:
        steps_since_complete = self.curr_frame_pos - self.completed_at
        if steps_since_complete >= COMPLETE_GRACE_STEPS:
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

    return logits


_original_add_attention_spy = AlignmentStreamAnalyzer._add_attention_spy
_hook_handles = []


def _patched_add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
    def attention_forward_hook(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            step_attention = output[1].cpu()
            self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]

    target_layer = tfmr.layers[layer_idx].self_attn
    handle = target_layer.register_forward_hook(attention_forward_hook)
    _hook_handles.append(handle)

    if hasattr(tfmr, 'config') and hasattr(tfmr.config, 'output_attentions'):
        self.original_output_attentions = tfmr.config.output_attentions
        self.original_attn_implementation = getattr(tfmr.config, '_attn_implementation', None)
        if getattr(tfmr.config, '_attn_implementation', None) == 'sdpa':
            tfmr.config._attn_implementation = 'eager'
        tfmr.config.output_attentions = True


def _clear_stale_hooks():
    while _hook_handles:
        _hook_handles.pop().remove()


_original_generate = ChatterboxMultilingualTTS.generate


def _patched_generate(self, *args, **kwargs):
    _clear_stale_hooks()
    return _original_generate(self, *args, **kwargs)


def apply():
    AlignmentStreamAnalyzer.step = _patched_step
    AlignmentStreamAnalyzer._add_attention_spy = _patched_add_attention_spy
    ChatterboxMultilingualTTS.generate = _patched_generate


def remove():
    AlignmentStreamAnalyzer.step = _original_step
    AlignmentStreamAnalyzer._add_attention_spy = _original_add_attention_spy
    ChatterboxMultilingualTTS.generate = _original_generate
