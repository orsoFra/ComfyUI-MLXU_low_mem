"""Fail-closed classification of a safetensors checkpoint's quantization format.

Classifies by MARKER KEYS (companion `.scale_weight`/`.input_scale`/`.comfy_quant`
tensors), never by dtype alone -- two files that are both "F8_E4M3" can need
completely different dequantization math depending on which scale convention
accompanies them. An unrecognized convention is reported as `Unrecognized`,
never silently treated as plain-castable: `_load_safetensors()` (native/__init__.py)
already upcasts naive FP8 correctly, but naively upcasting a *scaled* FP8
checkpoint through that same path would decode to noise, not raise an error --
exactly the silent-corruption failure mode this project has hit before with the
string/tuple checkpoint-key bug.

No diffusion-transformer checkpoint on this machine uses `scaled_fp8`/`fp4`/
`int8` -- only FP8_NAIVE is confirmed there (`darkBeast_dbkleinv2BFS.safetensors`,
see commit c6a4724). A real `.comfy_quant`-marked file does exist on this
machine (`qwen_3_8b_fp8mixed.safetensors`, a Flux.2 text encoder -- never
routed through `_load_safetensors`, so irrelevant to the loader gate itself),
and it mixes F8_E4M3- and U8-payload `.comfy_quant` weights in the same file;
testing against it is what caught the classifier's original bug where a mixed
file was confidently misclassified as FP4_PACKED (fixed: any dtype mix under
`.comfy_quant` now falls through to Unrecognized). FP8_SCALED and the
single-dtype-uniform INT8_TENSORWISE/FP4_PACKED branches remain covered by
synthetic-header tests only; treat them as unverified against a uniform real
file until one surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .safetensors_header import SafetensorsHeader


class QuantFormat(str, Enum):
    DENSE = "dense"                        # F16/BF16/F32 only, cast as-is
    FP8_NAIVE = "fp8_naive"                # F8_E4M3/F8_E5M2, no scale companion -- upcast is correct
    FP8_SCALED = "fp8_scaled"              # F8_E4M3/F8_E5M2 + *.scale_weight/*.input_scale sibling
    FP4_PACKED = "fp4_packed"              # *.comfy_quant marker, U8 nibble-packed payload
    INT8_TENSORWISE = "int8_tensorwise"    # *.comfy_quant marker, real I8 payload
    UNKNOWN = "unknown"                    # no recognized pattern -- never a guessed default


_DENSE_DTYPES = {"F16", "BF16", "F32"}
_FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}

# Non-weight index/buffer tensors that can appear in a full checkpoint
# alongside real weights but are never a quantized weight payload in any
# ComfyUI convention (those always use F8_E4M3/F8_E5M2/U8/I8 -- see the
# module docstring). Confirmed via a real SDXL checkpoint
# (fucktasticAnimeCheckpoint_22.safetensors:
# conditioner.embedders.0.transformer.text_model.embeddings.position_ids,
# shape (1, 77), I64 -- HF CLIP's standard position-ids buffer). Ignoring
# these dtypes for classification purposes only; `_load_safetensors` still
# loads the tensor itself unchanged, it just doesn't participate in the
# quant-format verdict.
_IGNORED_DTYPES = {"I64"}


@dataclass(frozen=True)
class Unrecognized:
    reason: str


def classify_quant_format(header: SafetensorsHeader) -> QuantFormat | Unrecognized:
    """Classify a checkpoint's quantization convention from its header alone.

    Order matters: `.comfy_quant`-marked tensors are checked first (they carry
    their own dtype signal independent of any FP8 tensors elsewhere in the same
    file), then FP8 scale-companion detection, then a plain-dense check. Any
    dtype not covered by one of these rules falls through to `Unrecognized`
    with the offending keys named, rather than being defaulted to DENSE.
    """
    dtypes_seen = {entry.dtype for entry in header.tensors.values()} - _IGNORED_DTYPES

    comfy_quant_keys = [k for k in header.tensors if k.endswith(".comfy_quant")]
    if comfy_quant_keys:
        # The quantized weight lives at the same prefix with .comfy_quant
        # stripped; its own dtype (U8 nibble-packed vs. real I8) decides the
        # convention -- the marker key alone is ambiguous between the two.
        # A real ComfyUI export can mix conventions PER LAYER within one file
        # (confirmed against qwen_3_8b_fp8mixed.safetensors: 141 .comfy_quant
        # weights in F8_E4M3, 85 in U8, same file) -- picking whichever dtype
        # happens to appear first would be exactly the overconfident-guess
        # failure mode this classifier exists to avoid. Only return a verdict
        # when EVERY .comfy_quant-marked weight agrees on one recognized
        # packed dtype; any mix, or a dtype outside {I8, U8}, is Unrecognized.
        packed_dtypes = set()
        for marker_key in comfy_quant_keys:
            prefix = marker_key[: -len(".comfy_quant")]
            weight_key = f"{prefix}.weight"
            if weight_key in header.tensors:
                packed_dtypes.add(header.tensors[weight_key].dtype)
        if packed_dtypes == {"I8"}:
            return QuantFormat.INT8_TENSORWISE
        if packed_dtypes == {"U8"}:
            return QuantFormat.FP4_PACKED
        return Unrecognized(
            f"found {len(comfy_quant_keys)} '.comfy_quant' marker(s) with "
            f"inconsistent or unrecognized payload dtype(s) {sorted(packed_dtypes)} "
            f"-- expected every marker to agree on exactly I8 or exactly U8"
        )

    if dtypes_seen & _FP8_DTYPES:
        has_scale_companion = any(
            k.endswith(".scale_weight") or k.endswith(".input_scale")
            for k in header.tensors
        )
        if has_scale_companion:
            return QuantFormat.FP8_SCALED
        return QuantFormat.FP8_NAIVE

    if dtypes_seen and dtypes_seen <= _DENSE_DTYPES:
        return QuantFormat.DENSE

    unrecognized_dtypes = dtypes_seen - _DENSE_DTYPES - _FP8_DTYPES
    return Unrecognized(
        f"unrecognized tensor dtype(s) in checkpoint: {sorted(unrecognized_dtypes)}"
    )
