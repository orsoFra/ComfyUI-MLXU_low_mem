"""State dict weight mapping for Flux2/Klein checkpoints.

Verified against a real checkpoint on this machine (`flux2Klein_9b.safetensors`,
201 top-level key patterns) — the checkpoint's own naming is already almost
exactly our internal naming, needing only the SAME three adjustments as
FLUX.1's weight_map.py (`native/weight_map.py`):

  double_blocks.{i}.img_mlp.0/2.weight      (GLU-gated MLP, still Sequential-indexed)
  double_blocks.{i}.txt_mlp.0/2.weight
  double_blocks.{i}.{img,txt}_attn.{qkv,proj}.weight     (no .bias — Flux2 is bias-free)
  double_blocks.{i}.{img,txt}_attn.norm.{query,key}_norm.scale
  single_blocks.{i}.linear1/linear2.weight
  single_blocks.{i}.norm.{query,key}_norm.scale
  double_stream_modulation_img/txt.lin.weight     (top-level, matches our attr names 1:1)
  single_stream_modulation.lin.weight             (top-level, matches our attr names 1:1)
  img_in.weight, txt_in.weight                    (no .bias)
  time_in.in_layer/out_layer.weight               (no .bias)
  final_layer.linear.weight, final_layer.adaLN_modulation.1.weight

No `vector_in.*` or `guidance_in.*` keys exist in this checkpoint (no pooled
conditioning, no guidance embedding) — nothing to map for those, and our
Flux2Transformer has no corresponding attributes at all (unlike FLUX.1,
which always allocates them and conditionally uses them).
"""

from __future__ import annotations

import mlx.core as mx


def normalize_flux2_keys(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Strip common ComfyUI/diffusion-model prefixes from checkpoint keys."""
    normalized: dict[str, mx.array] = {}
    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value
    return normalized


def map_flux2_to_native(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map a normalized Flux2/Klein state dict to our module naming.

    Identical three rules to FLUX.1's `map_flux_to_native` — verified this
    is not a coincidence but a consequence of both sharing the same
    `comfy.ldm.flux.model.Flux` base class and checkpoint-key conventions.
    """
    result: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        new_key = key

        # ── img_mlp.{0,2} / txt_mlp.{0,2} -> img_mlp_0/2, txt_mlp_0/2 ──────
        if ".img_mlp." in new_key:
            new_key = new_key.replace(".img_mlp.", ".img_mlp_")
        elif ".txt_mlp." in new_key:
            new_key = new_key.replace(".txt_mlp.", ".txt_mlp_")

        # ── QKNorm scale -> weight (RMSNorm convention) ─────────────────
        elif new_key.endswith(("query_norm.scale", "key_norm.scale")):
            new_key = new_key[: -len("scale")] + "weight"

        # ── final_layer.adaLN_modulation.{1} -> insert MLX Sequential .layers ──
        elif new_key.startswith("final_layer.adaLN_modulation."):
            suffix = new_key[len("final_layer.adaLN_modulation."):]
            new_key = f"final_layer.adaLN_modulation.layers.{suffix}"

        result[new_key] = value

    return result
