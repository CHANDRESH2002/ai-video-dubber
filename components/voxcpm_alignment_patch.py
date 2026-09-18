"""
Fix for VoxCPM2's own version of the Chatterbox stop-token bug: generation
sometimes fails to stop when real content is done, producing a trailing
low-level "murmur" (not silence -- confirmed via energy-profile analysis
and VAD, which classifies it as continuous speech) that can run for many
extra seconds. Directly instrumented and confirmed via
components/../scratchpad experiments (see chat history / the published
"VoxCPM2 Dataflow" artifact): stop_head's own predicted stop-probability
stays confidently near 0.0000 for ~10+ steps past where real content ends,
before eventually correcting -- the same failure shape documented for
Chatterbox in chatterbox_patch.py.

Root cause and fix, same technique as Chatterbox's AlignmentStreamAnalyzer
(chatterbox/models/t3/inference/alignment_stream_analyzer.py), ported to
VoxCPM2's different architecture:

  1. VoxCPM2 uses the fused scaled_dot_product_attention kernel, which
     never materializes attention weights -- forward_step is reimplemented
     manually (same math, causal mask, GQA head-repeat) for 4 specific
     (layer, head) pairs found by brute-force search across all 28 layers
     x 16 heads to show real text-tracking behavior: (18,11), (21,13),
     (22,9), (24,0). This is empirical, model-specific, exactly like
     Chatterbox's own hardcoded LLAMA_ALIGNED_HEADS = [(12,15),(13,11),(9,2)].

  2. The 4 heads are averaged, then argmax is restricted to ONLY the real
     text-token columns (excludes the reference-audio prefix and any
     already-generated audio positions) -- mirrors Chatterbox's
     `aligned_attn[:, i:j]` slicing.

  3. A two-consecutive-reading warm-up gate: the very first raw reading is
     compared against an arbitrary starting point (0) and would otherwise
     be accepted unconditionally, letting a noisy cold-start fluke lock in
     as the tracked position (confirmed via direct testing -- caused a
     genuine 0.48s premature cutoff on a real 5s+ English sentence before
     this guard was added). Only start tracking once two consecutive raw
     readings agree with each other.

  4. Once started, a discontinuity filter rejects single-step jumps outside
     (-15, +35) before updating the smoothed position -- Chatterbox's own
     bounds (-4, +7) are tuned for its much finer per-token stepping and
     were confirmed (via direct A/B testing) to freeze VoxCPM2's tracking
     too early, since one VoxCPM2 step (0.16s of audio) typically covers
     many text tokens' worth of progress at once.

  5. Completion fires once the smoothed position reaches within 3 of the
     end of the real text (S-3, same margin Chatterbox uses), then a fixed
     2-step grace period (calibrated for VoxCPM2's coarser ~0.16s/step,
     vs. Chatterbox's finer per-token COMPLETE_GRACE_STEPS=20) before
     forcing a stop -- regardless of what stop_head itself says.

Validated on 6 diverse test sentences (short/medium/long, Hindi/English,
styled/plain), 3 repeats each, 18 total generations: zero catastrophic
rambling, zero premature cutoffs, after the warm-up-gate fix above.

Apply once per process via voxcpm_alignment_patch.apply(model), after
VoxCPM.from_pretrained() and before any model.generate() call.
"""
import math
import types

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

CANDIDATE_HEADS = [(18, 11), (21, 13), (22, 9), (24, 0)]
DISCONTINUITY_BOUNDS = (-15, 35)
WARMUP_AGREEMENT_TOLERANCE = 10
COMPLETION_MARGIN = 3
GRACE_STEPS = 2

_step_head_weights: dict = {}


def _patched_attn_forward_step(self, hidden_states, position_emb, position_id, kv_cache):
    """Manual re-implementation of MiniCPMAttention.forward_step's math,
    identical output to the fused SDPA call it replaces, but also captures
    the softmax attention weights for this (layer, head) so the alignment
    tracker above can read them."""
    bsz, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_emb is not None:
        from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb
        cos, sin = position_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

    key_cache, value_cache = kv_cache
    key_cache[:, :, position_id, :] = k
    value_cache[:, :, position_id, :] = v

    num_groups = self.num_key_value_groups
    k_rep = key_cache.repeat_interleave(num_groups, dim=1)
    v_rep = value_cache.repeat_interleave(num_groups, dim=1)

    scores = (q.float() @ k_rep.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
    mask = (torch.arange(key_cache.size(2), device=key_cache.device) <= position_id)
    scores = scores.masked_fill(~mask.view(1, 1, 1, -1), float("-inf"))
    weights = F.softmax(scores, dim=-1)

    head = dict(CANDIDATE_HEADS)[self.layer_idx]
    _step_head_weights[self.layer_idx] = weights[0, head, 0, :].detach().cpu()

    attn_output = (weights.to(v_rep.dtype) @ v_rep).transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
    return self.o_proj(attn_output)


def _find_text_slice(text_mask_list: list[int]) -> tuple[int, int]:
    """The real content-text region is the trailing contiguous run of 1s in
    text_mask (excludes the reference-audio prefix, whose text_mask has
    isolated 1s at its start/end with zeros for the audio in between)."""
    ones = [i for i, v in enumerate(text_mask_list) if v == 1]
    j = ones[-1] + 1
    i = ones[-1]
    for pos in reversed(ones[:-1]):
        if pos == i - 1:
            i = pos
        else:
            break
    return i, j


@torch.inference_mode()
def _patched_inference(self, text, text_mask, feat, feat_mask, min_len=2, max_len=2000,
                        inference_timesteps=10, cfg_value=2.0, streaming=False, streaming_prefix_len=4):
    """Reimplementation of VoxCPM2Model._inference (non-streaming path
    only -- this project's pipeline never uses streaming mode), adding the
    alignment-based stopping rule alongside the original stop_head check.
    Whichever fires first ends generation."""
    if streaming:
        raise NotImplementedError("voxcpm_alignment_patch only covers the non-streaming path")

    text_mask_list = text_mask[0].cpu().numpy().tolist()
    i_bound, j_bound = _find_text_slice(text_mask_list)
    S = j_bound - i_bound

    B, T, P, D = feat.shape
    prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
    feat_embed = prefill_encoder(feat)
    feat_embed = self.enc_to_lm_proj(feat_embed)
    scale_emb = self.config.lm_config.scale_emb if self.config.lm_config.use_mup else 1.0
    text_embed = self.base_lm.embed_tokens(text) * scale_emb
    combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

    prefix_feat_cond = feat[:, -1, ...]
    pred_feat_seq = []
    context_len = 0  # reference-only mode (this project's only usage) always starts fresh

    enc_outputs, kv_cache_tuple = self.base_lm(inputs_embeds=combined_embed, is_causal=True)
    self.base_lm.kv_cache.fill_caches(kv_cache_tuple)
    enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
    lm_hidden = enc_outputs[:, -1, :]

    residual_enc_inputs = self.fusion_concat_proj(torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1))
    residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(inputs_embeds=residual_enc_inputs, is_causal=True)
    self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
    residual_hidden = residual_enc_outputs[:, -1, :]

    text_position = 0
    started = False
    prev_raw = None
    complete = False
    completed_at = None

    for step in tqdm(range(max_len)):
        dit_hidden = torch.cat((self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(residual_hidden)), dim=-1)
        pred_feat = self.feat_decoder(
            mu=dit_hidden, patch_size=self.patch_size,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=inference_timesteps, cfg_value=cfg_value,
        ).transpose(1, 2)

        curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))
        curr_embed = self.enc_to_lm_proj(curr_embed)
        pred_feat_seq.append(pred_feat.unsqueeze(1))
        prefix_feat_cond = pred_feat

        # _step_head_weights holds whatever the LAST base_lm.forward_step()
        # call produced -- same one-step lag stop_head itself has (it also
        # checks lm_hidden from the previous forward_step call, not this
        # iteration's new one).
        available = [_step_head_weights[l] for l, h in CANDIDATE_HEADS if l in _step_head_weights]
        if available:
            avg_weights = torch.stack(available).mean(dim=0)
            text_col_weights = avg_weights[i_bound:j_bound]
            cur_text_posn = int(text_col_weights.argmax().item())
            if not started:
                if prev_raw is not None and abs(cur_text_posn - prev_raw) < WARMUP_AGREEMENT_TOLERANCE:
                    started = True
                    text_position = cur_text_posn
                prev_raw = cur_text_posn
            else:
                lo, hi = DISCONTINUITY_BOUNDS
                discontinuity = not (lo < (cur_text_posn - text_position) < hi)
                if not discontinuity:
                    text_position = cur_text_posn
            complete = complete or (started and text_position >= S - COMPLETION_MARGIN)
            if complete and completed_at is None:
                completed_at = step

        stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
        stophead_says_stop = (step > min_len and stop_flag == 1)
        alignment_says_stop = complete and completed_at is not None and (step - completed_at) >= GRACE_STEPS

        if stophead_says_stop or alignment_says_stop:
            break

        lm_hidden = self.base_lm.forward_step(
            curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
        ).clone()
        lm_hidden = self.fsq_layer(lm_hidden)
        curr_residual_input = self.fusion_concat_proj(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
        residual_hidden = self.residual_lm.forward_step(
            curr_residual_input, torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device)
        ).clone()

    pred_feat_seq = torch.cat(pred_feat_seq, dim=1)
    feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
    generated_feat = pred_feat_seq[:, context_len:, :, :].squeeze(0).cpu()
    yield feat_pred, generated_feat, context_len


def apply(model) -> None:
    """model: a loaded VoxCPM instance (VoxCPM.from_pretrained(...)).
    Patches its underlying tts_model in place. Idempotent."""
    tts = model.tts_model
    layer_idxs = {l for l, h in CANDIDATE_HEADS}
    for layer in tts.base_lm.layers:
        if layer.self_attn.layer_idx in layer_idxs:
            layer.self_attn.forward_step = types.MethodType(_patched_attn_forward_step, layer.self_attn)
    tts._inference = types.MethodType(_patched_inference, tts)
