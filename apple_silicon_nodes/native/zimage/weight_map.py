"""State dict weight mapping for Z-Image (NextDiT) checkpoints.

Verified against a real checkpoint on this machine
(`unet/ZImageBase/base model/z_image_bf16.safetensors`) — a "diffusion
models only" split file (no `model.diffusion_model.` prefix, matching
FLUX/Krea2's distribution format), 453 keys.

Almost every key already matches our native module tree 1:1 — this project's
`native/zimage/model.py` was deliberately named to mirror the checkpoint's
own attribute names (`attention.qkv`, `attention.q_norm`/`k_norm`,
`feed_forward.w1/w2/w3`, `attention_norm1/2`, `ffn_norm1/2`, `cap_pad_token`,
`x_pad_token`, `x_embedder`, `final_layer.linear`, ...) — the only
transformations needed are `.layers.` insertions for the three MLX
`nn.Sequential` blocks:

  - `cap_embedder.{0,1}.*` (checkpoint) -> `cap_embedder.layers.{0,1}.*`
    (RMSNorm + Linear, no wrapper level difference — just needs the
    `.layers.` MLX Sequential convention).
  - `t_embedder.mlp.{0,2}.*` -> `t_embedder.layers.{0,2}.*` — the reference
    wraps its Sequential in an intermediate `TimestepEmbedder.mlp`
    attribute; our model skips that wrapper class entirely and makes
    `t_embedder` directly the Sequential, so this also strips `.mlp.`.
  - `{layers,noise_refiner}.{i}.adaLN_modulation.0.*` and
    `final_layer.adaLN_modulation.1.*` -> insert `.layers.` after
    `adaLN_modulation.` — covers all three modulation sites in one rule
    since the substring `adaLN_modulation.` never appears elsewhere.
    (`context_refiner` has no `adaLN_modulation` at all — `modulation=False`.)
"""

from __future__ import annotations

import mlx.core as mx


def normalize_zimage_keys(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Strip common ComfyUI/diffusion-model prefixes from checkpoint keys."""
    normalized: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value

    return normalized


def map_zimage_to_native(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map a normalized Z-Image state dict to our module naming."""
    result: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("cap_embedder."):
            new_key = "cap_embedder.layers." + new_key[len("cap_embedder."):]
        elif new_key.startswith("t_embedder.mlp."):
            new_key = "t_embedder.layers." + new_key[len("t_embedder.mlp."):]
        elif "adaLN_modulation." in new_key:
            new_key = new_key.replace("adaLN_modulation.", "adaLN_modulation.layers.")

        result[new_key] = value

    return result
