# Canon

Settled engineering decisions for this repository. The protocol lives in
`.claude/rules/canon.md`. Append new records at the end; supersede, never
rewrite. Managed with Claudoscope, safe to edit by hand.

## Kontext ref_latents implements only the single-reference case
kind: choice | date: 2026-08-04 | status: canon
For a single Kontext reference latent, comfy's three `ref_latents_method` variants
("index", "uxo", and the default "offset") all reduce to the same placement: RoPE
axis-0 index=1, h_offset=w_offset=0 — the h/w offset logic only matters for stacking
multiple references without spatial overlap. Because: this project's Kontext node
(`kontext_reference_latent`) only ever supplies one reference latent, so porting the
full multi-reference stacking/offset logic from `comfy/ldm/flux/model.py::_forward`
would be speculative complexity with no current caller. `FluxTransformer.get_rope`
already accepts a `ref_grids` list for future extension, but `sampler/core.py` only
ever builds a single-element list.

## MLX-to-PyTorch latent handoff needs an explicit MPS sync
kind: gotcha | date: 2026-08-04 | status: non-canon, superseded by: ASDX_VAEDecode's _fallback_decode double-indexed the latent
This record was wrong. The `IndexError` it describes was mis-diagnosed as an MLX/
PyTorch-MPS GPU race; the fix applied (`bridge.py::_mlx_to_torch_owned()`,
`torch.mps.synchronize()` + `.contiguous()` at every latent handoff) was reverted
after the real cause was found (see the superseding record) — it never reproduced
the bug against the actual buggy code path, only against a bypassed one. Kept here,
marked non-canon, so a future session doesn't re-reach for GPU-sync as the answer
to an intermittent-looking indexing error without checking the call arguments first.

## ASDX_VAEDecode's _fallback_decode double-indexed the latent
kind: gotcha | date: 2026-08-04 | status: canon
`ASDX_VAEDecode.decode()` already unwraps `latent = samples["samples"]` (a tensor)
before calling `_fallback_decode(latent, vae)`, but `_fallback_decode`'s body did
`vae.decode(latent["samples"])` — indexing the tensor with the string "samples" a
second time. Because: PyTorch's advanced indexing treats a bare string as a
non-tuple sequence of characters (`"samples"` has 7 chars), producing exactly
`IndexError: too many indices for tensor of dimension 4` on a 4D latent — and only
`--use-split-cross-attention`'s code path happened to surface it as a hard error in
this project's testing, which is what made it initially look like a GPU-interop
race (see the superseded record above) rather than a typo one line away. Fixed by
changing `_fallback_decode` to `vae.decode(latent)` — verified deterministically
(3/3 clean runs before the fix failed 3/3, 3/3 after the fix passed 3/3) against the
real Illustrious (SDXL) checkpoint with `--force-fp16 --use-split-cross-attention`,
the exact flags ComfyUI Desktop launches with on this machine. Lesson: when an error
looks intermittent or environment-specific, re-derive the repro from the ACTUAL
calling code path (the buggy node's own methods) before reaching for a GPU/timing
explanation — an early repro that bypassed `_fallback_decode` entirely (called
`vae.decode()` directly) is what made this look racy for longer than it should have.

## Node package migrated fully to ComfyUI's V3 API
kind: choice | date: 2026-08-09 | status: canon
All ~22 ASDX_* nodes use `io.ComfyNode` with `define_schema()` returning `io.Schema`
and `execute()` as a classmethod returning `io.NodeOutput(...)`, registered via a
single `ComfyExtension`/`comfy_entrypoint()` in `__init__.py` — not the V1
`NODE_CLASS_MAPPINGS`/`INPUT_TYPES()`/instance-method style. Because: V3 gives typed
I/O, per-node `fingerprint_inputs`/`validate_inputs` as first-class classmethods (used
on `ASDX_MemoryProfiler`/`ASDX_CacheManager`/`ASDX_LivePreview` to force
re-execution on side-effecting nodes instead of silently serving a stale cached
result), and the migration was verified end-to-end against the real ComfyUI V3
runtime (`comfy_api.latest`, via the installed ComfyUI's own venv) — every node's
`define_schema()` and `GET_NODE_INFO_V1()` conversion was exercised, not just
`py_compile`d. Every `node_id` was kept identical to its old V1 mapping key, so
existing saved workflows keep resolving to the same nodes. The old `_WEB_DIRECTORY
= "web"` dead variable (wrong name, no `web/` dir) was dropped in the same pass.

## Krea2T enhancer's ~x75 amplification needs a finite-value guard and bfloat16
kind: gotcha | date: 2026-08-09 | status: canon
`krea2t_enhance_conditioning` (`native/krea2/model.py`) multiplies chunks of the
stacked Qwen3-VL taps by up to 5x (per-chunk gain) times 15x (global multiplier) =
~75x before re-running `txtfusion` on the amplified copy. At `strength=1.0` (the
default, matching the reference node) this can overflow float16 (max ~65504) on
real Qwen3-VL hidden-state outliers that already reach into the thousands; the
resulting NaN/Inf in `candidate_out` propagated through `post_delta`/`token_scale`
with no guard, silently corrupting the final fused embedding to 100% NaN --
sampling and VAE decode still "succeeded" but produced a black image, with the
corruption invisible until `encode_text`'s output (confirmed via targeted
`[ASDX][DEBUG]` stat prints at 4 pipeline checkpoints: clean input, 100%-NaN
output right after `encode_text`). Fixed by (1) a finite-check in
`krea2t_enhance_conditioning` that falls back to the unamplified `reference_out`
if `candidate_out` isn't finite, and (2) recommending bfloat16 precision
(float32-range exponent, no overflow at these magnitudes) for full-strength
enhancer use -- confirmed no overflow warning at `precision=bfloat16` vs. a
reproducible one at `precision=float16`, same checkpoint/prompt/seed.

## CLIP `clip_type` must be set manually outside ASDX_CheckpointLoader
kind: gotcha | date: 2026-08-09 | status: canon
Only `ASDX_CheckpointLoader` auto-detects the text-encoder architecture from the
checkpoint's own embedded CLIP weights (`comfy.sd.load_checkpoint_guess_config`).
The standalone `ASDX_CLIPLoader`/`ASDX_DualCLIPLoader` nodes require the user to
pick the right `clip_type` value themselves; leaving it at a generic default (not
"krea2") silently loads a plain single-layer Qwen3-VL encoder instead of
`Krea2TEModel` (`comfy/text_encoders/krea2.py`, 12-layer tap, 30720-dim fused
output) -- this project's own `bridge.py::conditioning_krea2_to_mlx` then falls
back to tiling that single layer 12x (the `"repeating single-layer embedding"`
warning), discarding most of the real conditioning signal. Result: a valid,
non-NaN image that doesn't match the prompt at all, not a crash -- easy to miss.
Same class of footgun already flagged for Klein-4B in `loader.py`; confirmed here
to also apply to Krea2.
