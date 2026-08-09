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
