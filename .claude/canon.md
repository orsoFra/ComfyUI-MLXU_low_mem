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

## LoRA merge OOM has two independent duplication sources, not one
kind: gotcha | date: 2026-08-12 | status: canon
`lora.py::_load_lora_file` eagerly computes `delta = B @ A` (a full `[out,in]`
reconstruction) for EVERY LoRA target at load time, before any merge starts --
for a densely-targeted LoRA this alone approaches a near-full-size second copy
of every touched weight. `_apply_lora_to_transformer` then builds a SECOND full
parameter set via `type(transformer)(config)` + `.update(tree_unflatten(...))`.
Both structures are alive simultaneously during the merge loop, on top of the
original (referenced) base weights -- explains the observed ~2x/98GB peaks on a
~44GB model. Because: two earlier fix attempts (chunked `mx.eval()`, then +
`mx.clear_cache()` between chunks -- still present in `_apply_lora_to_transformer`)
had zero effect because neither addressed either structure's *lifetime*, only
transient per-chunk scratch -- a future session hitting the same OOM symptom
should look at what's held alive across the whole merge, not just what's
evaluated per iteration. Source #1 (eager delta materialization) was fixed this
session -- see the next record. Source #2 (whole-model rebuild) is unfixed;
the target architecture for it is the next record's forward-time residual
pattern (Phase 1-3, not started).

## LoRA target architecture is mlx-gen's forward-time residual (AdaptableLinear), ported in phases; Phase 0 done
kind: choice | date: 2026-08-12 | status: non-canon, superseded by: LoRA forward-time-residual architecture: Phases 1-3 done (Krea2, FLUX.1/Flux.2, Z-Image); SDXL stays on merge
`inference/crates/media/mlx-gen/src/adapters.rs`'s pattern -- an `AdaptableLinear`
wrapping `base: LinearBase` + `adapters: Vec<Adapter>`, where `Adapter::residual(x)`
adds each LoRA/LoKr's forward-time contribution WITHOUT ever merging into or
copying the base weight (`base(x) + Σ adapter.residual(x)`) -- is the chosen
long-term fix for duplication source #2 above (SDXL is the one deliberate
exception, kept on an in-place merge for ancestral-sampler bit-exactness, per
that same crate). Phased: **Phase 0 (done this session, `lora.py`)** --
`_load_lora_file` no longer eagerly computes `delta = B @ A` for every target;
it stores the tiny raw `(A, B)` factors in `LoRAAdapter.factors` instead, and
`_apply_lora_to_transformer`/the diffusers-key resolvers now materialize each
target's full-size delta lazily, one at a time, right where it's already
consumed by the existing chunked eval/`clear_cache` loop -- eliminating
duplication source #1 without touching the merge architecture itself. Phase 1
(proof-of-concept residual wrapper on Krea2 -- simplest, no fused qkv keys),
Phase 2 (extend to FLUX.1/FLUX.2's fused `qkv`/`linear1` via multi-adapter
residual slices), Phase 3 (Z-Image) are NOT started and not yet approved --
this record exists so a future session doesn't have to re-derive the mlx-gen
reference pattern from scratch before resuming.

## comfy's automatic tiled-VAE OOM fallback never fires on Apple Silicon MPS
kind: gotcha | date: 2026-08-12 | status: canon
`comfy.sd.VAE.decode()`/`.encode()` already retry via tiled decode
(`decode_tiled_`) when `model_management.is_oom(e)` recognizes the caught
exception as OOM -- but `is_oom()` only matches `torch.cuda.OutOfMemoryError`
or `torch.AcceleratorError` (error_code==2 / "out of memory" in message). A
real MPS OOM on this machine (torch 2.13) raises a plain `RuntimeError`
("MPS backend out of memory...", confirmed by direct reproduction), which is
NOT an instance of either type (`AcceleratorError` is a RuntimeError subclass,
not the reverse) -- so `is_oom()` returns `False`, `raise_non_oom()` re-raises,
and the whole node crashes instead of falling back to tiled decode. Because:
this is a real, unfixed, MPS-only gap directly in the jetsam/OOM-crash theme
of this project's memory work, distinct from the LoRA-merge duplication above
-- but it lives in `ASDX_VAEDecode`/`ASDX_VAEEncode`'s own `_fallback_decode`/
`_fallback_encode` (`vae.py`), not in `comfy/model_management.py` (out of this
repo's scope to patch upstream). Not yet implemented: wrap the `vae.decode`/
`vae.encode` call in `vae.py` with a `RuntimeError` + `"out of memory"` message
check that calls `vae.decode_tiled`/`vae.encode_tiled` explicitly, rather than
relying on comfy core's OOM detection. Porting `mlx-gen`'s own `vae_tiling.rs`
does NOT apply here -- that module tiles a native MLX VAE decode graph, and
this project deliberately has none (see "Porting VAE encode/decode to MLX has
negative ROI" above); the two tiling problems are architecturally unrelated.

## LoRA forward-time-residual architecture: Phases 1-3 done (Krea2, FLUX.1/Flux.2, Z-Image); SDXL stays on merge
kind: choice | date: 2026-08-12 | status: non-canon, superseded by: full-size LoRA deltas are merged once into `.weight`, never held as a residual
All four non-SDXL native families now apply LoRA via `AdaptableLinear`
(`lora.py`) instead of `_apply_lora_to_transformer`'s merge:
`_apply_lora_residual_to_krea2` (Phase 1, `SingleStreamDiT.blocks`),
`_apply_lora_residual_to_flux` (Phase 2, shared by `FluxTransformer` and
`Flux2Transformer`'s `double_blocks`/`single_blocks`),
`_apply_lora_residual_to_zimage` (Phase 3, `NextDiT`'s three block lists
`context_refiner`/`noise_refiner`/`layers`). `_apply_lora_to_transformer`
dispatches to these by `isinstance`
before reaching its own merge body, which is now SDXL-only (dead code for
the FLUX diffusers-fallback variables was removed accordingly). Mechanism,
shared via `_adapt_leaf`/`_clone_block_path`: each attach call shallow-
clones (`copy.copy`, `Module` is a `dict` subclass) only the object-graph
path from the transformer root down to each touched `nn.Linear` -- never
`type(transformer)(config)` -- and wraps that one `Linear` as an
`AdaptableLinear` holding `_lora_factors`/`_lora_deltas` (leading
underscore, so `Module.valid_parameter_filter` excludes them from
`.parameters()`/checkpoint save/`tree_flatten`). A native (comfy/BFL-
format) LoRA target is a whole fused weight already, so it attaches as a
raw `(A, B)` pair with ZERO full-size materialization. A diffusers/PEFT-
trained LoRA that splits a fused `qkv`/`linear1` into separate `to_q`/
`to_k`/`to_v` factors can't be represented as one low-rank residual (each
has its own rank) -- for FLUX.1/Flux.2 those are still assembled into one
full-width delta via the EXISTING `_resolve_flux_diffusers_lora`/
`_resolve_flux2_diffusers_lora`/`_assemble_fused_delta` (unchanged since
Phase 0) and attached as a single `_lora_deltas` residual instead --
materializing that one weight, never the whole model. Because: this
eliminates BOTH duplication sources in the two records above for every
family except SDXL, which intentionally keeps the old merge (ancestral-
sampler bit-exactness, per the mlx-gen crate's own rationale). Two real
bugs were found and fixed while porting, both worth knowing before
touching this code again: (1) FLUX.1's per-block modulation Linears
(`img_mod.lin`/`txt_mod.lin`/`modulation.lin`) were missing from the
residual target list at first -- `_FLUX_DOUBLE_DIFFUSERS_RENAME`/
`_FLUX_SINGLE_DIFFUSERS_RENAME` already proved these are real, confirmed
LoRA targets, so a hand-maintained per-family target tuple must be cross-
checked against those rename tables, not just the qkv/mlp keys that seem
obvious; (2) see the next record for the `ASDX_LoraSchedule` accumulation
bug -- a separate, more subtle trap in the same porting work. Verified so
far only by synthetic numerical tests (non-destructive cloning, reference-
sharing for untouched blocks, residual-vs-explicit-merge agreement within
float32 rounding, chained multi-LoRA safety, parameter-tree exclusion) --
NOT yet verified against a real checkpoint + real LoRA file + real
generation in ComfyUI, which is the necessary next step before trusting
this in production.

## `AdaptableLinear` adapters must upsert by array identity, not append, or `ASDX_LoraSchedule` grows unbounded
kind: gotcha | date: 2026-08-12 | status: canon
`sampler/core.py::_update_lora_schedule` re-applies the LoRA every sampling
step with `lora.scale` temporarily set to `delta_scale = new_scale -
scale_prev` (not the absolute new scale), relying on the OLD merge path's
`value = value + delta*delta_scale` to converge on `orig_value +
delta*new_scale` across repeated calls without compounding -- see that
function's own docstring. A naive port of this to the residual path (`leaf.
_lora_factors.append((a, b, lora.scale))` on every attach call) would
instead append one MORE redundant adapter entry to the touched leaf on
EVERY step, since each call sees the same `(a, b)` (from `lora.factors`,
loaded once, never recreated) but the naive code has no way to recognize
"this is the same adapter, not a new one" -- the forward pass would then
re-run one more low-rank matmul per step per touched leaf, unbounded over
the whole sampling run. Fixed by `_upsert_lora_factor`/`_upsert_lora_delta`
(`lora.py`): find an existing entry with the same `a`/`b` array IDENTITY
and ADD the incoming scale to its stored scale (not replace, not append) --
`scale_prev + delta_scale == new_scale` reproduces the merge path's exact
accumulation contract with zero changes to `sampler/core.py`. Because: this
was caught only by writing a synthetic test that explicitly simulates 10
schedule steps and asserts the adapter list length stays at 1 -- the
single-application tests (load once, generate once) all passed fine and
would never have caught it, so any future adapter-carrying LoRA mechanism
in this codebase needs a repeated-reapplication test, not just a one-shot
one.

## Auditing the residual LoRA port against real files found real coverage gaps synthetic tests couldn't
kind: gotcha | date: 2026-08-13 | status: canon
After Phases 1-3 (see the two records above) were implemented and passed
synthetic tests, running `_load_lora_file` against this machine's actual
LoRA library (`/Volumes/MBP2021/Images/models/loras/`) surfaced THREE real,
previously-undetected coverage gaps -- all now fixed in `lora.py`:
(1) **kohya-ss flat naming** (`lora_unet_double_blocks_0_img_attn_proj.
lora_up.weight` etc.) is what MOST real community-trained FLUX.1 LoRAs on
this machine actually use (24/24 sampled Flux.1-D files, 0 native-dotted) --
the residual attach functions only checked the dotted native key directly,
so this alone would have silently applied zero deltas for nearly this
project's entire real FLUX.1 library. Fixed by `_lookup_native_or_kohya`
(shared by all three phases), which tries the dotted key then the kohya
reconstruction (built FROM the known key, forward, same non-ambiguity
reasoning as the pre-existing merge-path fallback it mirrors).
(2) **Krea2's `txtfusion` sub-transformer** (`layerwise_blocks`/
`refiner_blocks`, its own two `TextFusionBlock` lists, plus a standalone
`projector` Linear) was entirely unreachable by Phase 1's attach loop, which
only walked the main `blocks.*` list. Confirmed against real files that the
MAJORITY of this project's Krea2 character/clothing LoRAs target ONLY
`txtfusion.*` and never touch `blocks.*` at all -- this was not an edge
case, it was most of the library. Also added Krea2's other top-level
Linears (`first`, `last.linear`) for the same reason (found via a LoRA that
targets everything: `krea2_turbo_lora_rank_64_bf16.safetensors`).
(3) **FLUX.1/Flux.2's top-level embedders/output layer** (`img_in`,
`txt_in`, `time_in`/`vector_in`/`guidance_in` (`MLPEmbedder`, in_layer/
out_layer), `final_layer.linear`, and Flux.2's THREE global `Modulation`
instances -- `double_stream_modulation_img`/`_txt`/`single_stream_
modulation`, all outside any block list since Flux.2 computes modulation
once globally, not per-block) were also unreachable. Found via a real file
(`Flux_2-Turbo-LoRA_comfyui.safetensors`) that targets ONLY these three
global modulation Linears plus `final_layer.linear` -- zero block-level
keys at all; that file went from 0/170 to 170/170 resolved after the fix.
Because: none of this was caught by the Phase 1-3 synthetic tests, because
those tests constructed their OWN small fake transformers and OWN synthetic
`LoRAAdapter` targeting only the keys the author already knew to test --
a synthetic test proves the ATTACH MECHANISM is correct, it cannot prove
TARGET COVERAGE is complete, since an incomplete target list still passes
every test written against it. Only loading real, independently-authored
LoRA files and checking `len(lora.factors) + len(lora.deltas)` against how
many keys the attach function actually consumes closed this gap. A future
session adding LoRA support for a new architecture should budget time for
this real-file audit step, not stop at synthetic tests passing.

Two SEPARATE gaps were found during the same audit that are **NOT** fixed
and **NOT** caused by this porting work -- they predate it and affect the
old merge path identically, confirmed by checking whether the OLD merge's
plain `tree_flatten` key lookup would have matched either: (a)
Krea2's `tmlp`/`tproj`/`txtmlp` are `nn.Sequential`-wrapped, so their real
native keys are `tmlp.layers.0.weight` etc. (see `map_krea2_to_native`'s
`.layers.` insertion, applied only at CHECKPOINT load, never at LoRA load)
-- but real Krea2 LoRA files use the flat checkpoint-style `tmlp.0.weight`,
which matches neither the dotted-native nor the kohya-flat form, so this
was ALREADY silently broken before Phase 0. (b) ~9 real Krea2 LoRA files
(`krea2_softwatercolor.safetensors` and siblings) and most of this
machine's real Z-Image LoRA library use genuine HF diffusers-style keys
(`transformer.text_fusion.layerwise_blocks.0.attn.to_q.lora_A.weight` for
Krea2; `layers.0.attention.to_out.0.weight`/`layers.0.adaLN_modulation.0.
weight` for Z-Image) -- neither family has ever had a diffusers-key
resolver in this codebase (unlike FLUX.1/Flux.2's `_resolve_flux_diffusers_
lora`/`_resolve_flux2_diffusers_lora`), so these LoRAs applied zero deltas
under the old merge path too. Fixing either is new capability work (a
Krea2/Z-Image diffusers resolver, or a Krea2 Sequential-aware LoRA key
mapper), not a residual-porting fix -- out of scope for Phases 0-3, flagged
here so a future session doesn't have to rediscover it from a bug report.

## Full-size LoRA deltas are merged once into `.weight`, never held as a residual
kind: gotcha | date: 2026-08-13 | status: canon
The initial Phase 1-3 residual design (see the superseded record above)
stored EVERY resolved LoRA target -- both genuine low-rank `(A, B)` pairs
AND already-full-size `[out, in]` deltas (ComfyUI `.diff`/`.diff_b`
entries, and a diffusers/PEFT LoRA's multi-component-assembled delta from
`_resolve_flux_diffusers_lora`/`_resolve_flux2_diffusers_lora`) -- as a
forward-time residual (`AdaptableLinear._lora_deltas`, recomputed every
forward call, held resident for the model's entire lifetime). This was
WRONG for the full-size case: a real generation log (2026-08-13, FLUX.1
dev + a diffusers-trained LoRA, `Emma_Watson_c9fa.safetensors`, 494 raw
diffusers keys assembled into ~57 full-size fused-block deltas) showed peak
memory roughly DOUBLING (26.5GB with a native/kohya LoRA on the same
checkpoint -> 48.1GB with the diffusers one) and per-step latency
degrading catastrophically -- 4.1s/step (native) -> 22.7s/step (diffusers,
same generation) -> 127.7s/step (a SECOND generation reusing the identical
already-attached cached model, no re-attach in between). The escalation
across two runs of the IDENTICAL cached model -- not just a flat 2x slower
-- is consistent with MPS/unified-memory pressure from several GB of
full-size deltas held permanently resident (not just extra FLOPs, which
would predict a stable-but-slower per-step time, not a worsening one).
Because: a full-size delta gains NOTHING from the residual treatment (it
isn't cheap like a low-rank pair, so keeping it "unmerged" only pays cost
forever instead of once) and every family/mechanism has an already-private,
freshly-cloned leaf available to merge into by the time one is resolved
(see `_ensure_adaptable`). Fixed: `AdaptableLinear.merge_delta(delta,
scale)` does `self.weight = self.weight + scale * delta` ONCE, immediately,
instead of appending to a list -- `_lora_deltas` and its forward-pass loop
were removed from `AdaptableLinear` entirely; only genuine low-rank
`_lora_factors` remain as a true residual. Confirmed still compatible with
`ASDX_LoraSchedule`'s incremental `delta_scale` re-application trick (see
"`AdaptableLinear` adapters must upsert by array identity..." above) via a
synthetic test simulating 6 incremental `merge_delta` calls converging to
the same result as one full-scale merge (rel. err ~4e-7) -- `weight =
weight + delta*scale_prev` then `+= delta*delta_scale` sums to `weight +
delta*new_scale` exactly like the old merge path's own trick, no special
tracking needed since there's no list to keep bounded anymore. This bug
was invisible to every synthetic test written for Phases 1-3 (none of them
measured memory or wall-clock time, only correctness), and only surfaced
from a REAL generation with a REAL diffusers-heavy LoRA -- a reminder that
"numerically correct" and "same cost profile as what it replaced" are
different claims requiring different verification.

## Full-size LoRA merge-once fix confirmed in production, not just synthetically
kind: gotcha | date: 2026-08-13 | status: canon
The previous record's synthetic verification (`merge_delta` incremental-
scale test, rel. err ~4e-7) explicitly flagged real-world verification as
the outstanding step -- this closes that gap. Real ComfyUI generation logs
(2026-08-13, same `Emma_Watson_c9fa.safetensors` diffusers LoRA that
exposed the original bug) confirm the fix holds under real load: two
consecutive runs of the identical cached model+LoRA both show flat
~3.0-3.7s/step after an initial merge-cost first step (36.4s, then 6.9s on
the second run since the base checkpoint was already cached), and a stable
48.2GB peak on both runs -- no repeat of the prior 4.1s->22.7s->127.7s
escalation or memory doubling.

## ASDX_LoraSchedule's per-step re-application was never wired to any sampler -- the feature never worked for any family
kind: gotcha | date: 2026-08-13 | status: non-canon, superseded by: the SDXL schedule fix was incomplete -- Krea2/Z-Image/Flux2 had the identical wiring gap
Two independent, stacked bugs meant `ASDX_LoraSchedule`'s `strength_middle`/
`strength_end`/`strength_curve` had ZERO effect in every real generation
prior to this fix, for every model family, not just SDXL: (1) `_run_sdxl`
(`sampler/core.py`) never called `_update_lora_schedule` inside its step
loop at all -- only the shared FLUX/Krea2/Z-Image denoise loop did
(`sampler/core.py:297`), so SDXL always sampled at a frozen
`strength_start` regardless of the other schedule params. (2) More
fundamentally: `ASDX_LoraSchedule.execute()` (`lora.py:1796`) stores its
config in `model["lora_schedule"]`, but the sampler node
(`sampler/__init__.py`) never read that dict key -- it only accepted a
separate node input explicitly marked `# Legacy`
(`io.Custom("ASDX_LORA_SCHEDULE").Input(...)`), which NO node in this
codebase's schema outputs (`ASDX_LoraSchedule` only outputs
`asdx_model`/`mlx_clip`). So `self.lora_schedule` was `None` inside
`_SamplerCore` unconditionally, for every family -- bug (1) was
unreachable/moot until bug (2) was found, and fixing only `_run_sdxl`
(the first, more visible symptom) would still have produced silent no-op
scheduling. Both found from a single real generation log: the user's SDXL
test with `start=1.00 middle=1.00 end=0.50` showed the node's one-time
setup log line but never the per-step `"[ASDX] LoRA schedule: step N/28,
strength=..."` line that should print every 5 steps. Fixed by (a) wiring
`_update_lora_schedule` into `_run_sdxl`'s loop identically to the shared
denoise loop, and (b) in `sampler/__init__.py`:
`lora_schedule = model.get("lora_schedule") or lora_schedule`. Confirmed
against a real SDXL generation (same `DCinderella.safetensors` LoRA):
strength held at 1.000 through step 10 (progress <=0.5, start==middle),
then decayed 0.964->0.786->0.607 through steps 15/20/25, exactly matching
the linear curve toward `end=0.50`, with the expected added per-step cost
(full re-merge, ~0.70s->~1.05s/step) only once `delta_scale != 0`. Because:
this class of bug -- a data producer and consumer that silently never
connect, each individually "working" in isolation -- won't be caught by any
test that only exercises one side; only a real end-to-end log showing the
ABSENCE of an expected per-step log line surfaced it. Any future
"schedule"/"per-step" feature in this codebase should get an explicit
end-to-end smoke test (real node graph, checking for the per-step log
line), not just unit coverage of `_update_lora_schedule` in isolation.

## The SDXL schedule fix was incomplete -- Krea2/Z-Image/Flux2 had the identical wiring gap
kind: gotcha | date: 2026-08-13 | status: canon
The previous record's fix wired `_update_lora_schedule` into `_run_sdxl`
and mis-stated that "the shared FLUX/Krea2/Z-Image denoise loop" already
called it -- that was wrong. Only `run()` (FLUX.1's inline loop) had the
call; `_run_krea2`, `_run_zimage`, and `_run_flux2` (`sampler/core.py`)
were three MORE independent loops with the exact same missing wiring, only
discovered because the user retested the fix across every family, not just
SDXL: FLUX.1 and (now-fixed) SDXL showed the expected per-step
`"[ASDX] LoRA schedule: step N/..., strength=..."` lines, but Z-Image
showed none at all despite the node's setup line running, and Flux2's test
was inconclusive (interrupted by an unrelated latent-shape user error
before any step ran). Fixed by adding the identical
`if self.lora_schedule is not None: ...` block to all three loops. Because:
`_SamplerCore` has FIVE independent per-family sampling loops (`run()`,
`_run_krea2`, `_run_sdxl`, `_run_zimage`, `_run_flux2`), not one shared
loop with per-family branches -- any future cross-cutting per-step
concern (schedule, teacache-like caching, a new profiling hook) must be
checked/added against all five explicitly; grepping for the mechanism's
name across the whole file (not just the first loop found) is the way to
catch this, since a fix that "compiles and works for the family just
tested" gives no signal about the other four.

Two more findings surfaced by this same multi-family retest, neither
fixed, both worth knowing before relying on Schedule in production:
(1) **Krea2's precomputed text context ignores schedule changes.** Krea2's
loop precomputes `context = self.transformer.encode_text(...)` ONCE before
the step loop (to avoid a documented ~12s/step re-encode cost). Most real
Krea2 LoRAs target `txtfusion` only (see "Auditing the residual LoRA
port..." above) -- so even with the wiring fixed, a schedule on a
txtfusion-only LoRA changes `self.transformer` every step but has NO
visible effect after step 0, since `context` is never recomputed from the
updated transformer. Fixing this properly means re-encoding text per step,
which reintroduces the cost the precompute exists to avoid -- a real
trade-off, not a quick fix, left for a future decision.
(2) **Schedule is much more expensive on the residual families than on
SDXL's merge path.** A real FLUX.1 run (`Emma_Watson_c9fa.safetensors`,
494 factors, `start=1.00 middle=1.00 end=0.50`) showed step time jump from
~3s to ~13-14s (4-5x) and peak memory rise from ~48GB (unscheduled) to
68.7GB once `delta_scale != 0` started firing every step -- because each
call re-runs `_apply_lora_residual_to_flux`'s full non-destructive
clone-and-wrap traversal, not a simple in-place scale bump. SDXL's
equivalent merge-based re-apply only cost ~0.70s->~1.05s (1.5x) over the
same kind of window. Numerically still correct (same `_upsert_lora_factor`
identity-based accumulation as any other residual attach), just far more
expensive per step than the merge path for a densely-targeted LoRA -- worth
optimizing (e.g. mutate existing `_lora_factors` scale in place instead of
re-cloning the object graph) if Schedule + a large residual-family LoRA
becomes a common real workflow, but not done here.
