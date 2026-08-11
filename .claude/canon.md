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

## `resource.ru_maxrss` measures a loading-time peak, not steady-state memory
kind: gotcha | date: 2026-08-11 | status: canon
When profiling `comfy.sd.load_clip()`'s real CPU RAM cost (to evaluate porting the
text encoder to MLX), `resource.getrusage(...).ru_maxrss` looked ~2x the on-disk
fp16 weight size for every family (e.g. SDXL clip_l+clip_g: 3.10GB peak vs 1.52GB on
disk) and stayed inflated even after explicitly forcing `model_options={"dtype":
torch.float16}` — a result that looked like a persistent fp32-upcast bug. It wasn't:
`ru_maxrss` is a monotonic high-water mark for the whole process and never drops,
so it captures the transient buffer `comfy.sd.load_clip()` holds only during
`load_state_dict()` (the raw safetensors buffer alongside the model's own tensor
copies) and keeps reporting that peak forever after, even once GC has freed it.
Re-measuring live process RSS via `ps -o rss= -p <pid>` immediately after an
explicit `gc.collect()` showed the real steady-state cost is only ~2-8% over the
on-disk fp16 size for SDXL/FLUX.1/Z-Image/Krea2 — matching what
`text_encoder_dtype()` (`comfy/model_management.py:1209`) already promises by
default. Because: this project profiles memory on Apple Silicon a lot (`memory.py`,
`memory_calibration.py`) and `ru_maxrss`/PyTorch's own `.current_allocated_memory()`
peak-style counters are an easy trap for the same false-positive on any future
memory investigation — always cross-check a suspicious peak reading against live
RSS post-GC before concluding a framework/dtype is wasting memory.

## Porting CLIP/T5/Qwen text encoders to MLX has weak memory ROI, except FP8 sources
kind: choice | date: 2026-08-11 | status: canon
Measured on this machine with real checkpoints (see the `ru_maxrss` gotcha above for
methodology): live steady-state CPU RAM for `comfy.sd.load_clip()` is already within
2-8% of the on-disk fp16/bf16 weight size for SDXL (1.52GB disk -> 1.62GB live),
FLUX.1 (9.35GB -> 9.52GB), Z-Image (7.49GB -> 7.72GB), and Krea2 (8.27GB -> 8.52GB) --
against MLX backbone active-memory footprints of 4.78GB/22.17GB/11.46GB/24.49GB
respectively (measured via `mx.get_active_memory()` after `load_sdxl_unet`/
`load_transformer`/`load_zimage_transformer`/`load_krea2_transformer`). Because: on
Apple Silicon, PyTorch-CPU and MLX draw from the same physical Unified Memory pool --
there is no separate VRAM pool to "free" by switching frameworks, and the CPU text
encoder path is already running the dtype MLX would use. The one real, measured
exception is Flux.2/Klein's FP8 text encoder (Qwen3-8B, `Huihui-Qwen3-8B-abliterated-
v2-FP8.safetensors`): 7.63GB on disk but 15.79GB live (~2.07x, matching the FP8->fp16
byte-width ratio exactly) because CPU has no FP8 compute path and must eagerly
dequantize to fp16 before use. That gap is a quantization-handling gap (same class as
`native/weight_format.py::classify_quant_format`'s FP8_SCALED handling for the
diffusion backbone), not fundamentally an MLX-vs-PyTorch framework problem -- fixing
it doesn't require porting CLIP's architecture to MLX, only teaching the text-encoder
load path to dequantize FP8 the same way the backbone already does.

## Porting VAE encode/decode to MLX has negative ROI -- PyTorch-MPS kernels win
kind: choice | date: 2026-08-11 | status: canon
Unlike CLIP (see the record above), `comfy.sd.VAE` already runs on `device=mps`, not
CPU (confirmed on all 4 real VAE checkpoints tested: Z-Image, FLUX.1, Flux.2/Klein,
Krea2), so there's no CPU-vs-GPU gap to close and no CPU RAM duplication to save.
Real `vae.decode()` timing at 1024x1024 (3 runs, `torch.mps.synchronize()`-bracketed):
Z-Image 0.310s, FLUX.1 0.314s, Flux.2 0.317s (128ch/16x downscale) -- Krea2 1.139s,
~3.6x slower than the others, but that's the Wan21-style causal 3D VAE architecture
(`comfy/ldm/wan/vae.py`), not a framework effect. The MLX<->PyTorch bridge round-trip
at the VAE boundary (`bridge.py::mlx_to_comfy_image`'s `.cpu().numpy()`/`mx.array()`
direction, and the reverse for the input latent) measured under 1.1% of total decode
time for every family (~0.1ms latent-in, ~3ms image-out vs. 300-1140ms decode) -- ruling
out bridge overhead as a plausible target, contrary to the initial hypothesis going into
this test. A direct micro-benchmark of the VAE decoder's actual building block (conv2d +
GroupNorm + SiLU, bf16, MPS vs MLX, at the real channel/resolution combos a ResNet
decoder stage sees) showed PyTorch-MPS consistently faster than MLX, and the gap widens
at exactly the regime that dominates real decode time (high spatial res, few channels):
1.1x at 128x128/512ch, 1.5x at 256x256/256ch, 2.6x at 512x512/128ch, 4.2x at
1024x1024/64ch. Because: porting the VAE decoder (ResNet blocks, GroupNorm, attention,
plus Krea2's Wan21 causal 3D convs) to MLX would very likely make decode SLOWER, not
faster, for real engineering effort and no memory upside -- the opposite conclusion from
CLIP. `mlx_vae.py`'s untrained placeholder class should stay unused/dead code; this is
not a case of "finish porting it," it's a case of "don't."

## Flux.2/Klein's FP8 text encoder and a native-BF16 alternative cost the same live RAM
kind: choice | date: 2026-08-11 | status: canon
Tested the FP8 file's non-FP8 alternative once it became available locally
(`qwen3-8b-heretic.safetensors`, header-verified pure BF16 -- 399/399 tensors `BF16`,
zero `.scale_weight`/`.input_scale` marker keys, 15.26GB on disk) with the same live-RSS
methodology as the `resource.ru_maxrss` gotcha above: live steady-state RAM after
`gc.collect()` is 15.79GB -- statistically identical to the FP8 file's own measured
15.79GB live footprint (7.63GB on disk, ~2.07x from CPU-side dequant to fp16, see the
`Porting CLIP/T5/Qwen...` record above). Because: this is exactly what that record
predicted -- FP8's on-disk compactness is fully cancelled by the CPU dequant-to-fp16
cost, so it lands at the same live size as an equivalent-precision BF16 file that's
already ~2x larger on disk. There is no memory case for switching to a non-FP8 Flux.2
text encoder; the FP8 file is strictly better for disk footprint at equal RAM cost. Any
preference for `heretic` over the `Huihui` FP8 file would have to be about generation
quality/abliteration behavior, not memory -- and the real memory fix for Flux.2/Klein
stays the FP8->bf16 dequant-path implementation already identified, not a file swap.
