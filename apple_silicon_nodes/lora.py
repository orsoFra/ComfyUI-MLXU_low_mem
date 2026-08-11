"""
LoRA Runtime Loading
====================
Load and apply LoRA adapters to FLUX transformers at runtime.

Supports:
  - Standard LoRA (A/B matrices)
  - ComfyUI diff format (.diff, .diff_b)
  - Per-LoRA alpha scaling (alpha / rank)
  - Multiple LoRA stacking with individual weights
  - LoRA baking (fuse into base weights for speed)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch

from comfy_api.latest import io

from . import bridge
from .native import FluxTransformer
from .native.flux2 import Flux2Transformer
from .native.safetensors_header import read_safetensors_header
from .native.sdxl import UNetModel as SDXLUNetModel


# ── LoRA↔model family compatibility ─────────────────────────────────

@dataclass(frozen=True)
class LoRAFamilySignature:
    family: str  # a CapabilityProfile.family value, or "unknown"
    evidence: tuple[str, ...]


# Base architecture a CapabilityProfile.family value is built on, for
# compatibility comparison -- flux1_fill/flux1_depth are FLUX.1 variants that
# share the exact double_blocks/single_blocks key namespace a plain "flux1"
# LoRA targets, so they must not be flagged as a mismatch.
_LORA_COMPATIBLE_BASE: dict[str, str] = {
    "flux1": "flux1",
    "flux1_fill": "flux1",
    "flux1_depth": "flux1",
    "flux2": "flux2",
    "krea2": "krea2",
    "sdxl": "sdxl",
    "zimage": "zimage",
}

# Conservative, substring-based signatures over RAW (unstripped) safetensors
# header keys -- a real LoRA file's keys still carry whatever prefix
# (diffusion_model./model./transformer.) `_load_lora_file` strips later, but
# substring matching doesn't care about a leading prefix.
#
# Verified against real LoRA files on this machine for flux1 (BFL-native),
# flux2, krea2, sdxl (kohya-ss, the dominant convention for
# Illustrious/Pony/NoobAI in this project's library) and zimage. A
# diffusers/PEFT-style FLUX/Flux2 LoRA (`transformer_blocks.{i}.attn.to_q...`,
# `single_transformer_blocks.{i}...`) returns "unknown" here rather than a
# false match -- `_apply_lora_to_transformer` resolves that naming separately
# via `_resolve_flux_diffusers_lora`/`_resolve_flux2_diffusers_lora`, so this
# check only needs to not misclassify it, not parse it.
#
# `.double_blocks.` alone (not `.single_blocks.`) is the flux1 signal here,
# but it is NOT conclusive by itself: Flux.2/Klein reuses FLUX.1's exact
# `double_blocks`/`single_blocks`/`img_attn`/`linear1`/`linear2` naming (both
# share the same `comfy.ldm.flux.model.Flux` class), so a genuine Flux.2 LoRA
# also carries `.double_blocks.` keys (confirmed on a real Civitai Flux.2
# Klein 9B LoRA). Flux.2's own `double_stream_modulation_img./_txt.` marker
# is used as a positive signal when present, but real-world LoRA files
# rarely train the shared top-level Modulation layers, so it's usually
# absent -- `_check_lora_compatibility` cross-checks actual block COUNTS
# against the loaded model's config before refusing a flux1/flux2 mismatch
# found here, to catch this ambiguity.
def detect_lora_family(path: Path) -> LoRAFamilySignature:
    """Header-only (no tensor data) detection of which base family a LoRA
    file targets, from distinctive key fragments. Zero or multiple matching
    signatures both return family="unknown" -- never a forced guess; callers
    should only refuse on a confident, single, DISAGREEING match, not on
    "unknown" (see `_check_lora_compatibility`).
    """
    header = read_safetensors_header(path)
    keys = header.tensors.keys()

    matches: dict[str, str] = {}  # family -> the key fragment that matched
    if any(".double_blocks." in k for k in keys):
        matches["flux1"] = "double_blocks."
    if any(".input_blocks." in k for k in keys) or any("_input_blocks_" in k for k in keys):
        matches["sdxl"] = "input_blocks."
    if any(".noise_refiner." in k for k in keys) or any(
        ".layers." in k and ".attention.to_" in k for k in keys
    ):
        matches["zimage"] = "noise_refiner./layers.*.attention.to_*"
    if any(".double_stream_modulation_img." in k or ".double_stream_modulation_txt." in k for k in keys):
        matches["flux2"] = "double_stream_modulation_img./_txt."
    # Krea2's SingleStreamDiT names attention projections attn.wq/wk/wv/wo
    # directly under blocks.{i} -- distinct from FLUX's fused img_attn.qkv/
    # txt_attn.qkv, so this never collides with the flux1 signature above.
    if any(".attn.wq." in k or ".attn.wk." in k or ".attn.wv." in k for k in keys):
        matches["krea2"] = "attn.wq/wk/wv"

    if len(matches) != 1:
        return LoRAFamilySignature(family="unknown", evidence=tuple(sorted(matches)))
    family, evidence = next(iter(matches.items()))
    return LoRAFamilySignature(family=family, evidence=(evidence,))


def _lora_block_counts(path: Path) -> tuple[int, int]:
    """Count of distinct double_blocks./single_blocks. indices present in a
    LoRA file's header (BFL-native naming). Header-only, no tensor data read.
    """
    header = read_safetensors_header(path)
    double_idx = {int(m.group(1)) for k in header.tensors for m in [re.search(r"double_blocks\.(\d+)\.", k)] if m}
    single_idx = {int(m.group(1)) for k in header.tensors for m in [re.search(r"single_blocks\.(\d+)\.", k)] if m}
    return len(double_idx), len(single_idx)


def _check_lora_compatibility(lora_path: Path, model: dict) -> None:
    """Refuse to apply a LoRA whose detected family disagrees with the
    loaded model's family -- silently applying deltas keyed by coincidental
    name overlap (or applying 0 deltas with only a quiet log line) is the
    failure mode this guards against. An "unknown" detection never blocks:
    it means "no confident signature found", not "confirmed mismatch".
    """
    signature = detect_lora_family(lora_path)
    if signature.family == "unknown":
        return
    model_family = model["capability"].family
    model_base = _LORA_COMPATIBLE_BASE.get(model_family, model_family)
    lora_base = _LORA_COMPATIBLE_BASE.get(signature.family, signature.family)
    if lora_base != model_base:
        # flux1 and flux2/Klein share the exact `comfy.ldm.flux.model.Flux`
        # class and its double_blocks/single_blocks/img_attn/linear1/linear2
        # naming -- only block COUNT and hidden_size differ (Klein 9B: 8
        # double + 24 single vs FLUX.1-dev: 19 double + 38 single), so a
        # "flux1 evidence: double_blocks." detection can be a false positive
        # on a genuine Flux2 LoRA. Cross-check the LoRA's actual block counts
        # against the loaded model's real config before trusting the naming
        # alone (confirmed false positive on a real Civitai-labeled Flux.2
        # Klein 9B LoRA: 8 double + 24 single blocks in the file, exactly
        # matching the loaded Klein checkpoint).
        if {lora_base, model_base} == {"flux1", "flux2"}:
            lora_double, lora_single = _lora_block_counts(lora_path)
            config = model["config"]
            if (
                lora_double == getattr(config, "num_double_blocks", -1)
                and lora_single == getattr(config, "num_single_blocks", -1)
            ):
                return
        raise ValueError(
            f"ASDX: '{lora_path.name}' looks like a {signature.family} LoRA "
            f"(evidence: {', '.join(signature.evidence)}), but the loaded "
            f"model is {model_family}. Applying it would silently apply "
            f"deltas to unrelated layers. Refusing."
        )


# ── LoRA Target Definition ────────────────────────────────────────────

@dataclass(frozen=True)
class LoRATarget:
    """Defines which transformer layer a LoRA weight key maps to."""
    # Path through the transformer module tree
    path_parts: tuple[str, ...]
    # Weight key suffixes in the LoRA file
    key_suffixes: tuple[str, ...]  # e.g. ("lora_A.weight", "lora_B.weight")


# FLUX.1 LoRA target patterns
_FLUX_LORA_TARGETS: tuple[LoRATarget, ...] = (
    # Double block attention
    LoRATarget(("double_blocks", "{i}", "img_attn"), ("qkv.weight",)),
    LoRATarget(("double_blocks", "{i}", "img_attn"), ("proj.weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_attn"), ("qkv.weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_attn"), ("proj.weight",)),
    # Double block MLP
    LoRATarget(("double_blocks", "{i}", "img_mlp_0"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "img_mlp_2"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_mlp_0"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_mlp_2"), ("weight",)),
    # Single block attention
    LoRATarget(("single_blocks", "{i}", "attn"), ("qkv.weight",)),
    LoRATarget(("single_blocks", "{i}", "attn"), ("proj.weight",)),
    # Single block MLP
    LoRATarget(("single_blocks", "{i}", "mlp_0"), ("weight",)),
    LoRATarget(("single_blocks", "{i}", "mlp_2"), ("weight",)),
)

# Krea2 (SingleStreamDiT) LoRA target patterns
# Keys in checkpoint: diffusion_model.blocks.{i}.attn.wq.lora_A.weight
# After stripping prefix: blocks.{i}.attn.wq
_KREA2_LORA_TARGETS: tuple[LoRATarget, ...] = (
    # Attention projections
    LoRATarget(("blocks", "{i}", "attn", "wq"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wk"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wv"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wo"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "gate_proj"), ("weight",)),
    # MLP projections
    LoRATarget(("blocks", "{i}", "mlp", "up"), ("weight",)),
    LoRATarget(("blocks", "{i}", "mlp", "gate"), ("weight",)),
    LoRATarget(("blocks", "{i}", "mlp", "down"), ("weight",)),
)


# ── LoRA Adapter ─────────────────────────────────────────────────────

@dataclass
class LoRAAdapter:
    """A single LoRA adapter with its weights and scale."""
    name: str
    # Map from (block_index, layer_type, param) -> delta weight
    deltas: dict[tuple[int, str, str], mx.array] = field(default_factory=dict)
    # None means the file has no ".alpha" key (see _load_lora_file) -- not
    # the same as alpha=1.0, the two fall back to different scales below.
    alpha: float | None = None
    rank: int = 0
    scale: float = 1.0

    def __post_init__(self):
        if self.rank == 0:
            # Infer rank from first delta
            if self.deltas:
                self.rank = next(iter(self.deltas.values())).shape[-1]
        if self.scale == 0:
            self.scale = base_lora_scale(self.alpha, self.rank)


def _normalize_native_lora_key(prefix: str) -> str:
    """Rename a stripped native BFL LoRA key prefix to match this project's
    live module tree, undoing the same checkpoint-convention renames
    `native/weight_map.py` applies at checkpoint-load time (see its own
    docstring) -- a LoRA trained against the real checkpoint's dotted
    `img_mlp.0`/`img_mlp.2` Sequential naming otherwise never matches the
    native module's flat `img_mlp_0`/`img_mlp_2` attributes, silently
    dropping every double-block MLP delta (confirmed: a real Flux.2 Klein
    LoRA lost exactly its 4 MLP keys/block, 32/112 deltas, this way).
    """
    if ".attn.gate." in prefix:
        prefix = prefix.replace(".attn.gate.", ".attn.gate_proj.")
    if ".img_mlp." in prefix:
        prefix = prefix.replace(".img_mlp.", ".img_mlp_")
    elif ".txt_mlp." in prefix:
        prefix = prefix.replace(".txt_mlp.", ".txt_mlp_")
    return prefix


def base_lora_scale(alpha: float | None, rank: int) -> float:
    """`alpha/rank` if the file declared its own alpha, else a flat 1.0 --
    matches comfy/weight_adapter/lora.py's fallback exactly (`alpha =
    v[2]/mat2.shape[0] if v[2] is not None else 1.0`). Do NOT default missing
    alpha to `1.0/rank` -- that under-applies the LoRA for any file that
    simply doesn't ship an alpha key.
    """
    return alpha / max(rank, 1) if alpha is not None else 1.0


# ── Diffusers/PEFT FLUX.1 / Flux.2 LoRA key mapping ─────────────────────
#
# A diffusers-trained FLUX.1 or Flux.2/Klein LoRA (e.g. ai-toolkit,
# SimpleTuner, kohya's sd-scripts diffusers mode) uses HF's
# `transformer_blocks`/`single_transformer_blocks` naming with separate
# `to_q`/`to_k`/`to_v` projections, not BFL's fused `double_blocks`/
# `single_blocks` `qkv`/`linear1`. Ported directly from comfy's own
# `comfy/utils.py::flux_to_diffusers` (the exact table comfy uses for this
# same file convention, and which already covers both families -- Flux.2's
# single-block entries are just extra keys appended to the same
# `block_map`) rather than guessed. comfy applies each split component as
# an independent sliced patch via `weight.narrow(dim, start, length)`
# because each of q/k/v/mlp keeps its own (possibly different) LoRA rank
# and can't be concatenated at the A/B stage -- `_assemble_fused_delta`
# below reproduces that by zero-padding each already-computed (B @ A)
# delta to the fused weight's full output width and concatenating.
#
# Double-stream blocks (`transformer_blocks.{i}`) are handled identically
# for both families by `_resolve_flux_double_diffusers_lora`: FLUX.1 and
# Flux.2/Klein share the exact same `double_blocks` module tree (see
# `native/flux2/config.py`'s docstring), and comfy's own key_map uses one
# unmodified `block_map` for both. Single-stream blocks differ: FLUX.1's
# diffusers port keeps `to_q`/`to_k`/`to_v`/`proj_mlp` split (needs the
# same assembly as the double-block qkv), while Flux.2's diffusers port
# already fuses them into one `attn.to_qkv_mlp_proj` weight matching
# native `linear1` exactly (a straight rename, no assembly) and
# `attn.to_out` covers the full `linear2` -- see comfy's
# `flux_to_diffusers`, the `# Flux 2` entries in its single-block
# `block_map`. Confirmed against a real Civitai Flux.2 Klein 9B diffusers
# LoRA's actual warning keys (8 `transformer_blocks.*` + 24
# `single_transformer_blocks.*`, matching Klein 9B's block counts).
#
# Native attribute names (see native/__init__.py's DoubleBlock/SingleBlock,
# and native/flux2/model.py's DoubleBlock/SingleBlock) match their BFL
# checkpoint 1:1 EXCEPT the MLP layers, which are flat `img_mlp_0`/
# `img_mlp_2` attributes (underscore) instead of a dotted Sequential
# `img_mlp.0`/`img_mlp.2` -- see native/weight_map.py's own docstring for
# the same rename applied at checkpoint-load time.
_FLUX_DOUBLE_QKV_RE = re.compile(r"^double_blocks\.(\d+)\.(img_attn|txt_attn)\.qkv\.weight$")
_FLUX_SINGLE_LINEAR1_RE = re.compile(r"^single_blocks\.(\d+)\.linear1\.weight$")
_FLUX_DOUBLE_RENAME_RE = re.compile(r"^double_blocks\.(\d+)\.(.+)$")
_FLUX_SINGLE_RENAME_RE = re.compile(r"^single_blocks\.(\d+)\.(.+)$")

# Native MLP suffix -> candidate diffusers suffixes, tried in order. Two
# real-world naming variants exist for the same weight (diffusers' own
# `ff.net.0.proj`/`ff.net.2` vs. the LyCoris/LoKr export convention's
# `ff.linear_in`/`ff.linear_out`, seen on the actual Flux.2 Klein LoRA this
# was verified against) -- comfy's own key_map maps both onto the same
# native target, so both are tried here too.
_FLUX_DOUBLE_DIFFUSERS_RENAME: dict[str, tuple[str, ...]] = {
    "img_attn.proj.weight": ("attn.to_out.0.weight",),
    "txt_attn.proj.weight": ("attn.to_add_out.weight",),
    "img_mod.lin.weight": ("norm1.linear.weight",),
    "txt_mod.lin.weight": ("norm1_context.linear.weight",),
    "img_mlp_0.weight": ("ff.net.0.proj.weight", "ff.linear_in.weight"),
    "img_mlp_2.weight": ("ff.net.2.weight", "ff.linear_out.weight"),
    "txt_mlp_0.weight": ("ff_context.net.0.proj.weight", "ff_context.linear_in.weight"),
    "txt_mlp_2.weight": ("ff_context.net.2.weight", "ff_context.linear_out.weight"),
}
_FLUX_SINGLE_DIFFUSERS_RENAME: dict[str, str] = {
    "modulation.lin.weight": "norm.linear.weight",
    "linear2.weight": "proj_out.weight",
}


def _assemble_fused_delta(
    pieces: list[mx.array | None], lengths: list[int]
) -> tuple[mx.array | None, int]:
    """Concatenate disjoint per-slice LoRA deltas (q/k/v[/mlp]) into one
    fused-weight delta. `pieces`/`lengths` are in output-axis order and
    span the fused weight's full output width contiguously -- a missing
    slice (LoRA doesn't target that component) becomes zeros so the
    concatenation still lines up. Returns (delta, num_pieces_consumed) --
    the count lets the caller's applied/total log line reflect actual raw
    delta entries consumed, since one fused native key here can absorb
    multiple raw file entries (e.g. 3 for a qkv projection).
    """
    consumed = sum(p is not None for p in pieces)
    if consumed == 0:
        return None, 0
    in_features = next(p.shape[1] for p in pieces if p is not None)
    dtype = next(p.dtype for p in pieces if p is not None)
    blocks = [
        p.astype(mx.float32) if p is not None else mx.zeros((length, in_features), dtype=mx.float32)
        for p, length in zip(pieces, lengths)
    ]
    return mx.concatenate(blocks, axis=0).astype(dtype), consumed


def _resolve_flux_double_diffusers_lora(
    flat_key: str, deltas: dict[str, mx.array], hidden_dim: int
) -> tuple[mx.array | None, int]:
    """Resolve a native `double_blocks.*` flat parameter key against a
    diffusers/PEFT-style LoRA's already-computed per-component deltas.
    Shared by both FLUX.1 and Flux.2/Klein -- see module-level comment
    above. Returns (None, 0) if `flat_key` isn't a `double_blocks.*` key or
    has no diffusers counterpart in `deltas`.
    """
    m = _FLUX_DOUBLE_QKV_RE.match(flat_key)
    if m:
        idx, stream = m.group(1), m.group(2)
        sub_names = ("to_q", "to_k", "to_v") if stream == "img_attn" else \
            ("add_q_proj", "add_k_proj", "add_v_proj")
        pieces = [deltas.get(f"transformer_blocks.{idx}.attn.{name}.weight") for name in sub_names]
        return _assemble_fused_delta(pieces, [hidden_dim, hidden_dim, hidden_dim])

    m = _FLUX_DOUBLE_RENAME_RE.match(flat_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        for diffusers_suffix in _FLUX_DOUBLE_DIFFUSERS_RENAME.get(rest, ()):
            delta = deltas.get(f"transformer_blocks.{idx}.{diffusers_suffix}")
            if delta is not None:
                return delta, 1
        return None, 0

    return None, 0


def _resolve_flux_diffusers_lora(
    flat_key: str, deltas: dict[str, mx.array], hidden_dim: int, mlp_dim: int
) -> tuple[mx.array | None, int]:
    """Resolve a native FLUX.1 flat parameter key against a diffusers/PEFT-
    style LoRA's already-computed per-component deltas (see module-level
    comment above). Returns (None, 0) if `flat_key` has no diffusers
    counterpart in `deltas` -- callers fall through to leaving the
    parameter unchanged.
    """
    delta, consumed = _resolve_flux_double_diffusers_lora(flat_key, deltas, hidden_dim)
    if consumed:
        return delta, consumed

    m = _FLUX_SINGLE_LINEAR1_RE.match(flat_key)
    if m:
        idx = m.group(1)
        pieces = [
            deltas.get(f"single_transformer_blocks.{idx}.attn.to_q.weight"),
            deltas.get(f"single_transformer_blocks.{idx}.attn.to_k.weight"),
            deltas.get(f"single_transformer_blocks.{idx}.attn.to_v.weight"),
            deltas.get(f"single_transformer_blocks.{idx}.proj_mlp.weight"),
        ]
        return _assemble_fused_delta(pieces, [hidden_dim, hidden_dim, hidden_dim, mlp_dim])

    m = _FLUX_SINGLE_RENAME_RE.match(flat_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        diffusers_suffix = _FLUX_SINGLE_DIFFUSERS_RENAME.get(rest)
        if diffusers_suffix is not None:
            delta = deltas.get(f"single_transformer_blocks.{idx}.{diffusers_suffix}")
            return delta, (1 if delta is not None else 0)
        return None, 0

    return None, 0


def _resolve_flux2_diffusers_lora(
    flat_key: str, deltas: dict[str, mx.array], hidden_size: int
) -> tuple[mx.array | None, int]:
    """Resolve a native Flux.2/Klein flat parameter key against a
    diffusers/PEFT-style LoRA's already-computed per-component deltas (see
    module-level comment above). Double-stream blocks reuse
    `_resolve_flux_double_diffusers_lora`; single-stream blocks are direct
    renames (`attn.to_qkv_mlp_proj` -> `linear1`, `attn.to_out` ->
    `linear2`), unlike FLUX.1 which needs 4-way assembly for `linear1`.
    """
    delta, consumed = _resolve_flux_double_diffusers_lora(flat_key, deltas, hidden_size)
    if consumed:
        return delta, consumed

    m = _FLUX_SINGLE_LINEAR1_RE.match(flat_key)
    if m:
        idx = m.group(1)
        delta = deltas.get(f"single_transformer_blocks.{idx}.attn.to_qkv_mlp_proj.weight")
        return delta, (1 if delta is not None else 0)

    m = _FLUX_SINGLE_RENAME_RE.match(flat_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        if rest == "linear2.weight":
            delta = deltas.get(f"single_transformer_blocks.{idx}.attn.to_out.weight")
            return delta, (1 if delta is not None else 0)
        return None, 0

    return None, 0


def _apply_lora_to_clip(clip: Any, lora_path: Path, strength_clip: float) -> Any:
    """Patch a CLIP text encoder with a LoRA file.

    `mlx_clip` (see conditioning.py) is a real `comfy.sd.CLIP` -- a PyTorch/
    ModelPatcher object, not our own MLX reimplementation -- so unlike
    `_apply_lora_to_transformer` there is no reason to hand-roll key
    matching here. `comfy.sd.load_lora_for_models(model, clip, ...)` already
    does this correctly (alpha/rank normalization included); passing
    `model=None` skips the model-side patch we already handle ourselves in
    MLX. Needed for SDXL, where (unlike FLUX/Flux2/Krea2/Z-Image, whose
    LoRAs are trained on the transformer only) real CLIP-side LoRA deltas
    are common (kohya `lora_te1_`/`lora_te2_` keys) and silently dropping
    them under-applies the LoRA's intended effect.

    `raw` is filtered to drop transformer-only keys before the comfy call:
    with `model=None`, comfy's internal key_map only covers CLIP, so every
    transformer-side key in the file (already applied separately by
    `_apply_lora_to_transformer`) would otherwise log a spurious
    "lora key not loaded" warning -- for a real kohya SDXL LoRA (e.g.
    Illustrious) that is thousands of warning lines burying any genuine
    CLIP-side mismatch. Covers the kohya `lora_unet_*` convention, the
    diffusers/PEFT FLUX/Flux2 convention -- bare `transformer_blocks.*`/
    `single_transformer_blocks.*`, NOT wrapped in a `transformer.` prefix,
    confirmed against a real diffusers Flux.2 Klein LoRA whose raw keys
    have no prefix at all -- and the BFL-native convention
    (`diffusion_model.double_blocks./single_blocks.*`, confirmed to
    otherwise flood the log with one warning per raw tensor on a real
    Flux.2 Klein LoRA) -- real CLIP LoRA keys use `lora_te*_`/`text_model.`
    naming, never any of these prefixes, so this can't drop a genuine
    CLIP-side key.
    """
    if clip is None or strength_clip == 0:
        return clip
    import comfy.sd
    import comfy.utils

    raw = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
    raw = {k: v for k, v in raw.items()
           if not k.startswith("lora_unet_") and not k.startswith("transformer.")
           and not k.startswith("diffusion_model.")
           and not k.startswith("transformer_blocks.")
           and not k.startswith("single_transformer_blocks.")}
    _, new_clip = comfy.sd.load_lora_for_models(None, clip, raw, 0.0, strength_clip)
    return new_clip if new_clip is not None else clip


# ── LoRA Loader Node ─────────────────────────────────────────────────

class ASDX_LoraLoader(io.ComfyNode):
    """Load a LoRA adapter and apply it to a model.

    Supports standard LoRA (A/B matrices) and ComfyUI diff format.
    Multiple LoRAs can be stacked by chaining LoRA loaders.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_LoraLoader",
            display_name="🍏 ASDX LoRA Loader",
            category="ASDX/LoRA",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Combo.Input("lora_name", options=cls._get_loras()),
                io.Float.Input("strength_model", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Custom("mlx_clip").Input("clip", optional=True),
                io.Float.Input("strength_clip", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
                io.Custom("mlx_clip").Output(display_name="clip"),
            ],
        )

    @staticmethod
    def _get_loras() -> list[str]:
        """Get list of available LoRA files."""
        try:
            import folder_paths
            loras = []
            for folder in ("loras",):
                try:
                    loras.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if loras:
                return loras
        except Exception:
            pass
        return ["example_lora.safetensors"]

    @classmethod
    def execute(
        cls,
        model: dict,
        lora_name: str,
        strength_model: float,
        clip: Any = None,
        strength_clip: float = 1.0,
    ) -> io.NodeOutput:
        """Load and apply a LoRA adapter to the model (and, if connected, the CLIP)."""
        t0 = time.perf_counter()

        transformer = model["transformer"]
        lora_path = cls._resolve_lora_path(lora_name)
        _check_lora_compatibility(lora_path, model)

        # Load LoRA weights (alpha comes from the file itself, see
        # _load_lora_file -- matches real ComfyUI's LoraLoader, which has no
        # user-facing alpha widget either).
        lora = cls._load_lora_file(lora_path)

        # Apply scale
        lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_model

        # Apply to transformer — returns a NEW transformer, the cached base
        # model (model["transformer"]) is never mutated (see
        # _apply_lora_to_transformer docstring).
        new_transformer = cls._apply_lora_to_transformer(transformer, lora, model["config"])
        new_model = {**model, "transformer": new_transformer}

        # The old (now-orphaned) transformer's arrays are unreferenced after
        # this reassignment, but MLX's allocator won't return that memory to
        # the OS on its own -- trim it now instead of letting it sit as idle
        # cache through however many more nodes run before something else
        # happens to clear it (loader.py does the same after every load).
        bridge.clear_mlx_cache()

        # CLIP-side LoRA -- only meaningful when a clip is actually connected
        # (SDXL); see _apply_lora_to_clip. Left as None when no clip input is
        # wired, matching this node's pre-existing FLUX/Krea2/etc. workflows
        # that never connect one.
        new_clip = _apply_lora_to_clip(clip, lora_path, strength_clip)

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] LoRA '{lora_name}' applied: rank={lora.rank}, "
              f"scale={lora.scale:.4f}, {elapsed:.2f}s")

        return io.NodeOutput(new_model, new_clip)

    @staticmethod
    def _load_lora_file(path: Path) -> LoRAAdapter:
        """Load a LoRA file and extract delta weights."""
        t0 = time.perf_counter()
        name = path.stem

        if path.suffix == ".safetensors":
            import torch
            import safetensors.torch
            state = safetensors.torch.load_file(path, device="cpu")
            raw = {}
            for k, v in state.items():
                if v.dtype == torch.bfloat16:
                    v = v.float()
                raw[k] = v.cpu().numpy()
        elif path.suffix == ".pt" or path.suffix == ".bin":
            import torch
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict):
                raw = {k: v.numpy() for k, v in state.items()}
            else:
                raw = {}
        else:
            raise ValueError(f"Unsupported LoRA format: {path.suffix}")

        # Extract deltas from raw weights
        lora = LoRAAdapter(name=name)
        deltas: dict[str, tuple[mx.array, mx.array]] = {}  # key -> (A, B) or diff
        alpha_value: float | None = None

        for key, weight in raw.items():
            if key.endswith(".alpha"):
                # Real per-file training alpha (comfy/lora.py's convention:
                # alpha_name = "{x}.alpha", same prefix as lora_A/B/up/down
                # minus any .weight suffix). Every target in a real LoRA
                # file shares the same value in practice -- keep the first
                # one found rather than tracking one per target.
                if alpha_value is None:
                    try:
                        alpha_value = float(weight)
                    except (TypeError, ValueError):
                        pass
                continue

            weight_arr = mx.array(weight if isinstance(weight, mx.array) else weight)

            # Standard LoRA format: {prefix}.lora_A.{param} / {prefix}.lora_B.{param}
            if ".lora_A." in key:
                # PEFT (HF diffusers-trained) files insert an adapter name
                # between lora_A/B and the param, e.g. "...lora_A.default.
                # weight" instead of comfy/kohya's "...lora_A.weight" -- a
                # plain string replace would leave "default.weight" stuck to
                # the prefix and never match anything downstream (confirmed:
                # a real diffusers Flux.2 LoRA applied 0/144 deltas this
                # way). lora_A/B always target a weight matrix, so drop
                # whatever follows and hard-code ".weight" back on.
                prefix = key.partition(".lora_A.")[0] + ".weight"
                # Strip common ComfyUI prefixes (diffusion_model., model., etc.)
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                # Native module-tree renames (Krea2 attn.gate -> attn.gate_proj,
                # FLUX/Flux2 img_mlp./txt_mlp. -> img_mlp_/txt_mlp_) -- see
                # _normalize_native_lora_key.
                prefix = _normalize_native_lora_key(prefix)
                if prefix not in deltas:
                    deltas[prefix] = (None, None)
                deltas[prefix] = (weight_arr, deltas[prefix][1])
            elif ".lora_B." in key:
                prefix = key.partition(".lora_B.")[0] + ".weight"
                # Strip common ComfyUI prefixes
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                prefix = _normalize_native_lora_key(prefix)
                if prefix not in deltas:
                    deltas[prefix] = (deltas[prefix][0], None)
                deltas[prefix] = (deltas[prefix][0], weight_arr)
            elif ".lora_up." in key:
                # kohya-style format: {prefix}.lora_up.weight / {prefix}.lora_down.weight
                down_key = key.replace(".lora_up.", ".lora_down.")
                if down_key not in raw:
                    continue
                prefix = key.replace(".lora_up.", ".")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                prefix = _normalize_native_lora_key(prefix)
                down_raw = raw[down_key]
                down_arr = down_raw if isinstance(down_raw, mx.array) else mx.array(down_raw)
                # Route through the same (a, b) pairing dict as lora_A/lora_B so the
                # shared "delta = b @ a" conversion loop below handles it uniformly
                # (up=B [out,rank], down=A [rank,in] -- comfy/weight_adapter/lora.py
                # computes `diff = lora_up.weight @ lora_down.weight`, i.e. B @ A).
                deltas[prefix] = (down_arr, weight_arr)
            elif ".diff_b" in key:
                # ComfyUI diff format (bias delta). Real key is "{x}.diff_b" with
                # NO .weight/.bias suffix on x (comfy/lora.py maps it to
                # "{x}.bias") -- append .bias, don't just strip the suffix.
                diff_key = key.replace(".diff_b", ".bias")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if diff_key.startswith(pfx):
                        diff_key = diff_key[len(pfx):]
                        break
                diff_key = _normalize_native_lora_key(diff_key)
                lora.deltas[diff_key] = weight_arr
            elif ".diff" in key:
                # ComfyUI diff format (weight delta). Real key is "{x}.diff" with
                # NO .weight suffix on x (comfy/lora.py maps it to "{x}.weight")
                # -- append .weight, don't just strip the suffix.
                diff_key = key.replace(".diff", ".weight")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if diff_key.startswith(pfx):
                        diff_key = diff_key[len(pfx):]
                        break
                diff_key = _normalize_native_lora_key(diff_key)
                lora.deltas[diff_key] = weight_arr

        # Convert (A, B) pairs to delta = B @ A and compute rank
        # lora_A is [rank, in_features], lora_B is [out_features, rank]
        for key, (a, b) in deltas.items():
            if a is not None and b is not None:
                if a.ndim == 4:
                    # Conv-style (LyCORIS/LoCon) LoRA: A keeps the full conv
                    # kernel [rank, in, kh, kw], B is the 1x1 channel-mixing
                    # projection [out, rank, 1, 1] -- a plain 2D matmul can't
                    # contract these directly (their trailing dims don't
                    # align), so flatten A's spatial extent into the matmul's
                    # contraction dim and reshape back to the conv weight
                    # shape afterwards.
                    rank, in_ch, kh, kw = a.shape
                    out_ch = b.shape[0]
                    a2d = a.reshape(rank, in_ch * kh * kw).astype(mx.float32)
                    b2d = b.reshape(out_ch, rank).astype(mx.float32)
                    delta = (b2d @ a2d).reshape(out_ch, in_ch, kh, kw)
                    # The checkpoint (and this raw LoRA tensor) is PyTorch
                    # [out, in, kh, kw], but every native MLX Conv2d parameter
                    # this delta gets added to is [out, kh, kw, in] (see
                    # native/sdxl/model.py's checkpoint-load transpose) --
                    # without this, the delta silently has the wrong shape
                    # for the target parameter it's summed into.
                    delta = delta.transpose(0, 2, 3, 1).astype(b.dtype)
                else:
                    # Standard LoRA: delta = B @ A
                    delta = (b.astype(mx.float32) @ a.astype(mx.float32)).astype(b.dtype)
                lora.deltas[key] = delta
                if lora.rank == 0:
                    lora.rank = a.shape[0]
            elif a is not None:
                lora.deltas[key] = a
                if lora.rank == 0:
                    lora.rank = a.shape[0]
            elif b is not None:
                lora.deltas[key] = b
                if lora.rank == 0:
                    lora.rank = b.shape[-1]

        # Clean up unused pairs
        lora.deltas = {k: v for k, v in lora.deltas.items()
                       if not (isinstance(v, tuple) and v[0] is None and v[1] is None)}

        lora.alpha = alpha_value
        mx.eval(*lora.deltas.values())
        return lora

    @staticmethod
    def _apply_lora_to_transformer(
        transformer: Any,
        lora: LoRAAdapter,
        config: Any,
    ) -> Any:
        """Apply LoRA delta weights to transformer parameters and return a
        NEW transformer instance — never mutates `transformer` in place.

        `transformer` may be the same object cached in loader.py's
        `_MODEL_CACHE` and shared across separate ComfyUI executions; an
        in-place `setattr` here would permanently bake the LoRA into that
        cached base model, so a later cache hit (e.g. re-running the same
        workflow after only changing an unrelated downstream widget) would
        re-apply the delta on top of an already-modified transformer and
        silently compound the LoRA's effect. Every sibling MLX project
        checked (mflux, SDMLX) applies LoRA non-destructively for this same
        reason.

        Untouched parameters are carried over by reference (no copy), so
        this is cheap relative to a real checkpoint reload — only the
        LoRA-targeted arrays are freshly computed.
        """
        from mlx.utils import tree_flatten, tree_unflatten

        if not lora.deltas:
            print("[ASDX] LoRA: no matching weights found")
            return transformer

        # key could be something like "double_blocks.0.img_attn.qkv.weight"
        # or "single_blocks.5.mlp_0.weight" — same dotted-string convention
        # tree_flatten uses for checkpoint keys (see native/*/model.py's
        # load_*_transformer), so deltas match model_flat keys directly.
        model_flat = tree_flatten(transformer.parameters())

        # Only SDXL's native module tree diverges from its real checkpoint
        # keys (MLX Sequential's `.layers.` insertion, see weight_map.py) --
        # FLUX/Krea2's native tree matches its checkpoint 1:1, so no un-
        # mapping is needed there (or if a future architecture needs one,
        # add it to that architecture's own weight_map.py, not here).
        # isinstance, not type(...).__module__ string matching: ComfyUI's custom
        # node loader imports this package under a name derived from its
        # install path (`nodes.py::load_custom_node`'s `sys_module_name =
        # module_path.replace(".", "_x_")`, using the FULL filesystem path, not
        # "apple_silicon_nodes") -- a `__module__` string comparison silently
        # never matches in a real ComfyUI install, only in a standalone script
        # importing this package by its repo name. isinstance is immune to
        # whatever name the outer package ends up loaded under, since the
        # imported class objects here and the ones used to build `transformer`
        # both resolve through the same relative-import chain.
        checkpoint_stem_fn = None
        if isinstance(transformer, SDXLUNetModel):
            from .native.sdxl.weight_map import native_key_to_checkpoint_stem
            checkpoint_stem_fn = native_key_to_checkpoint_stem

        # BFL-native FLUX.1 (native/__init__.py's FluxTransformer -- Krea2 reuses
        # this exact class, not flux2/sdxl/zimage which have their own) and
        # Flux.2/Klein (native/flux2/model.py's Flux2Transformer) are the only
        # families this diffusers/PEFT fallback covers -- see
        # `_resolve_flux_diffusers_lora`/`_resolve_flux2_diffusers_lora`'s
        # module comment.
        flux_diffusers_dims = None
        flux2_diffusers_hidden_size = None
        if isinstance(transformer, FluxTransformer):
            flux_diffusers_dims = (config.hidden_dim, config.mlp_dim)
        elif isinstance(transformer, Flux2Transformer):
            flux2_diffusers_hidden_size = config.hidden_size

        new_flat = []
        applied = 0
        for flat_key, value in model_flat:
            delta = lora.deltas.get(flat_key)
            if delta is None:
                # kohya-ss (sd-scripts) flat naming -- used by essentially all SD/
                # SDXL LoRAs (e.g. Illustrious): dots replaced by underscores and
                # prefixed "lora_unet_", e.g. "input_blocks.4.1.proj_in.weight" ->
                # "lora_unet_input_blocks_4_1_proj_in.weight". `_load_lora_file`
                # only strips a few known *dotted* prefixes, so these flat kohya
                # keys land in `lora.deltas` unchanged and never match `flat_key`
                # directly. Build the candidate the same direction comfy's real
                # `comfy/lora.py::model_lora_keys_unet` does (FROM the known real
                # key, not reverse-engineered from the flat one, which is
                # ambiguous whenever a module name itself contains an underscore,
                # e.g. "proj_in"/"ff_net_0_proj").
                #
                # `flat_key` is OUR native module's tree_flatten key, not the
                # real checkpoint key -- SDXL's weight_map.py inserts `.layers.`
                # (and restructures ff.net/label_emb) to address MLX's
                # nn.Sequential, so any target weight inside one of those
                # wrappers must be un-mapped back to its real checkpoint stem
                # before the kohya string is built, or it silently never
                # matches (this was the cause of the 442/986 partial match).
                stem, _, suffix = flat_key.rpartition(".")
                if checkpoint_stem_fn is not None:
                    stem = checkpoint_stem_fn(stem)
                delta = lora.deltas.get(f"lora_unet_{stem.replace('.', '_')}.{suffix}")
            consumed = 1 if delta is not None else 0
            if delta is None and flux_diffusers_dims is not None:
                # diffusers/PEFT FLUX.1 LoRA (transformer_blocks/
                # single_transformer_blocks, separate to_q/to_k/to_v) -- see
                # `_resolve_flux_diffusers_lora`.
                hidden_dim, mlp_dim = flux_diffusers_dims
                delta, consumed = _resolve_flux_diffusers_lora(flat_key, lora.deltas, hidden_dim, mlp_dim)
            if delta is None and flux2_diffusers_hidden_size is not None:
                # diffusers/PEFT Flux.2/Klein LoRA -- see
                # `_resolve_flux2_diffusers_lora`.
                delta, consumed = _resolve_flux2_diffusers_lora(
                    flat_key, lora.deltas, flux2_diffusers_hidden_size
                )
            if delta is not None:
                delta_mapped = delta.astype(value.dtype)
                new_flat.append((flat_key, value + delta_mapped * lora.scale))
                applied += consumed
            else:
                new_flat.append((flat_key, value))

        new_transformer = type(transformer)(config)
        new_transformer.update(tree_unflatten(new_flat))
        mx.eval(new_transformer.parameters())
        print(f"[ASDX] LoRA: applied {applied}/{len(lora.deltas)} deltas")
        return new_transformer

    @staticmethod
    def _resolve_lora_path(name: str) -> Path:
        """Resolve LoRA name to file path."""
        try:
            import folder_paths
            for folder in ("loras",):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "loras" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


# ── Multi LoRA Loader ────────────────────────────────────────────────

class ASDX_MultiLoraLoader(io.ComfyNode):
    """Load up to 5 LoRA adapters at once, each with its own on/off toggle
    and independent model/clip strengths.

    Functional parity with LoraManager's LoraLoaderLM and rgthree's Power
    Lora Loader (per-entry active toggle, separate strength_model/
    strength_clip) -- implemented with plain ComfyUI widgets rather than
    their dynamic add/remove JS/Vue frontends, since this project has no
    web/ frontend infrastructure at all.

    Applies all LoRAs in a single pass for better performance.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        loras = ASDX_LoraLoader._get_loras()
        inputs: list = [io.Custom("asdx_model").Input("model")]
        for i in range(1, 6):
            inputs.append(io.Boolean.Input(f"lora{i}_enabled", default=False))
            inputs.append(io.Combo.Input(f"lora{i}_name", options=loras))
            inputs.append(io.Float.Input(f"lora{i}_strength_model", default=1.0, min=-10.0, max=10.0, step=0.01))
            inputs.append(io.Float.Input(f"lora{i}_strength_clip", default=1.0, min=-10.0, max=10.0, step=0.01))
        inputs.append(io.Custom("mlx_clip").Input("clip", optional=True))
        return io.Schema(
            node_id="ASDX_MultiLoraLoader",
            display_name="🍏 ASDX Multi LoRA Loader",
            category="ASDX/LoRA",
            inputs=inputs,
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
                io.Custom("mlx_clip").Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        lora1_enabled: bool, lora1_name: str, lora1_strength_model: float, lora1_strength_clip: float,
        lora2_enabled: bool, lora2_name: str, lora2_strength_model: float, lora2_strength_clip: float,
        lora3_enabled: bool, lora3_name: str, lora3_strength_model: float, lora3_strength_clip: float,
        lora4_enabled: bool, lora4_name: str, lora4_strength_model: float, lora4_strength_clip: float,
        lora5_enabled: bool, lora5_name: str, lora5_strength_model: float, lora5_strength_clip: float,
        clip: Any = None,
    ) -> io.NodeOutput:
        """Apply up to 5 active LoRA adapters at once (and, if connected, the CLIP)."""
        entries = [
            (lora1_enabled, lora1_name, lora1_strength_model, lora1_strength_clip),
            (lora2_enabled, lora2_name, lora2_strength_model, lora2_strength_clip),
            (lora3_enabled, lora3_name, lora3_strength_model, lora3_strength_clip),
            (lora4_enabled, lora4_name, lora4_strength_model, lora4_strength_clip),
            (lora5_enabled, lora5_name, lora5_strength_model, lora5_strength_clip),
        ]

        transformer = model["transformer"]
        config = model["config"]
        applied_any = False

        for enabled, lora_name, strength_model, strength_clip in entries:
            if not enabled or not lora_name or lora_name == "None":
                continue
            if strength_model == 0 and strength_clip == 0:
                continue

            lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
            _check_lora_compatibility(lora_path, model)

            if strength_model != 0:
                lora = ASDX_LoraLoader._load_lora_file(lora_path)
                # Same alpha/rank normalization as ASDX_LoraLoader.load_lora,
                # so a rank-N LoRA doesn't apply at N times its intended
                # strength.
                lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_model

                # Reuse the (non-destructive, tree-flatten-based) apply
                # logic — do not duplicate it here. Each call returns a NEW
                # transformer; chaining `transformer =` here stacks LoRAs
                # onto that new object, never onto the cached base model.
                transformer = ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, config)
                # Each iteration orphans the previous intermediate
                # transformer (up to 5 full-size copies chained here) --
                # trim MLX's idle buffer cache between iterations instead of
                # letting them stack.
                bridge.clear_mlx_cache()

            clip = _apply_lora_to_clip(clip, lora_path, strength_clip)

            applied_any = True
            print(f"[ASDX] MultiLoRA: '{lora_name}' "
                  f"(model={strength_model:.2f}, clip={strength_clip:.2f})")

        if not applied_any:
            print("[ASDX] MultiLoRA: no LoRAs to apply")
            return io.NodeOutput(model, clip)

        return io.NodeOutput({**model, "transformer": transformer}, clip)


# ── LoRA Schedule (per-step strength modulation) ─────────────────────

class ASDX_LoraSchedule(io.ComfyNode):
    """Schedule LoRA strength across sampling steps.

    Allows LoRA strength to vary per step (e.g., stronger at start, weaker at end).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_LoraSchedule",
            display_name="🍏 ASDX LoRA Schedule",
            category="ASDX/Advanced",
            inputs=[
                io.Custom("asdx_model").Input("model"),
                io.Combo.Input("lora_name", options=ASDX_LoraLoader._get_loras()),
                io.Float.Input("strength_start", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("strength_end", default=0.5, min=0.0, max=10.0, step=0.01),
                io.Float.Input("strength_middle", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Combo.Input("strength_curve", options=["linear", "cosine", "ease_in_out"], default="linear"),
                io.Custom("mlx_clip").Input("clip", optional=True),
                io.Float.Input("strength_clip", default=1.0, min=0.0, max=10.0, step=0.01, optional=True),
            ],
            outputs=[
                io.Custom("asdx_model").Output(display_name="model"),
                io.Custom("mlx_clip").Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict,
        lora_name: str,
        strength_start: float,
        strength_end: float,
        strength_middle: float,
        strength_curve: str,
        clip: Any = None,
        strength_clip: float = 1.0,
    ) -> io.NodeOutput:
        """Attach LoRA schedule metadata to the model dict.

        The sampler reads this metadata and adjusts LoRA strength per step.
        """
        lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
        _check_lora_compatibility(lora_path, model)
        lora = ASDX_LoraLoader._load_lora_file(lora_path)

        # Apply with start strength (same alpha/rank normalization as
        # ASDX_LoraLoader.load_lora — a rank-N LoRA must not apply at N times
        # its intended strength). Returns a NEW transformer — the cached
        # base model is never mutated. `sampler/core.py::_update_lora_schedule`
        # then keeps adjusting THIS new (private, non-cached) transformer
        # in place per step, which is safe since it's no longer shared.
        lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_start
        new_transformer = ASDX_LoraLoader._apply_lora_to_transformer(
            model["transformer"], lora, model["config"]
        )
        bridge.clear_mlx_cache()
        new_model = {**model, "transformer": new_transformer}

        # CLIP-side LoRA is applied once at strength_clip (not scheduled):
        # text conditioning is computed once before the sampling loop, not
        # per step, so a step-varying CLIP strength has no sampler hook to
        # attach to -- see _apply_lora_to_clip.
        new_clip = _apply_lora_to_clip(clip, lora_path, strength_clip)

        # Store schedule info on the model — the sampler reads this and
        # adjusts LoRA strength per step (see sampler/core.py).
        new_model["lora_schedule"] = {
            "name": lora_name,
            "lora": lora,
            "strength_start": strength_start,
            "strength_end": strength_end,
            "strength_middle": strength_middle,
            "strength_curve": strength_curve,
        }

        print(f"[ASDX] LoRA schedule: '{lora_name}' "
              f"start={strength_start:.2f} middle={strength_middle:.2f} end={strength_end:.2f}")

        return io.NodeOutput(new_model, new_clip)

    @staticmethod
    def _compute_schedule_strength(
        step: int,
        total_steps: int,
        start: float,
        end: float,
        middle: float,
        curve: str,
    ) -> float:
        """Compute LoRA strength for a given step."""
        if total_steps <= 1:
            return start

        progress = step / total_steps

        if curve == "linear":
            # Start -> Middle (0-0.5) -> End (0.5-1.0)
            if progress <= 0.5:
                return start + (middle - start) * (progress * 2)
            else:
                return middle + (end - middle) * ((progress - 0.5) * 2)

        elif curve == "cosine":
            # Smooth cosine interpolation
            mid = (start + middle) / 2
            end_val = (middle + end) / 2
            if progress <= 0.5:
                return mid + (start - mid) * 0.5 * (1 - mx.cos(mx.array(progress * 2 * 3.14159)))
            else:
                return end_val + (end - end_val) * 0.5 * (1 - mx.cos(mx.array((progress - 0.5) * 2 * 3.14159)))

        elif curve == "ease_in_out":
            # S-curve: slow start, fast middle, slow end
            t = 3 * progress ** 2 - 2 * progress ** 3
            if progress <= 0.5:
                return start + (middle - start) * t
            else:
                return middle + (end - middle) * t

        return start  # default: constant


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_LIST = [
    ASDX_LoraLoader,
    ASDX_MultiLoraLoader,
    ASDX_LoraSchedule,
]
