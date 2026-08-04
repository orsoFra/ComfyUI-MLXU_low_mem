"""State dict weight mapping for SDXL UNet checkpoints.

Unlike FLUX, comfy does NOT rename SDXL's UNet internals — checkpoint keys
under `model.diffusion_model.` match `openaimodel.py::UNetModel`'s real
module tree almost 1:1 (`comfy/model_detection.py:1271-1291`). The only
adjustments needed:

  - Prefix strip (`model.diffusion_model.` / `diffusion_model.` / `model.`),
    same as FLUX/Krea2.
  - MLX's `nn.Sequential` stores children under a `.layers.` list attribute;
    PyTorch's `nn.Sequential` addresses them by flat index directly. Every
    Sequential block in our native/sdxl/model.py mirrors its PyTorch
    counterpart's child ordering exactly (including parameter-free SiLU/
    Dropout placeholders, so index numbers already line up) — the only
    transform needed is inserting `.layers.` after the attribute name:
    `time_embed.`, `in_layers.`, `emb_layers.`, `out_layers.`, `to_out.`,
    the top-level `out.`.
  - `label_emb` is DOUBLY nested in the real checkpoint
    (`nn.Sequential(nn.Sequential(Linear, SiLU, Linear))`, checkpoint keys
    `label_emb.0.0.*`/`label_emb.0.2.*`) but our native model flattens it to
    a single Sequential (the outer wrapper serves no purpose for SDXL) — so
    `label_emb.0.` collapses straight to `label_emb.layers.`.
  - `ff.net.{0,2}` (PyTorch: `BasicTransformerBlock.ff` is a `FeedForward`
    object whose own `.net` is the Sequential) maps to `ff.layers.{0,2}`
    (our native model skips the intermediate `FeedForward` wrapper and
    makes `ff` directly the Sequential).

Conv2d weights need a value-level transpose (PyTorch `[out,in,kh,kw]` ->
MLX `[out,kh,kw,in]`, since MLX convolutions are channel-last) — that is a
tensor-shape transform, not a key-string transform, so it is handled in
`model.py::load_sdxl_unet`'s weight-assignment loop, not here.
"""

from __future__ import annotations

import mlx.core as mx


def normalize_sdxl_keys(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Strip common ComfyUI/diffusion-model prefixes from checkpoint keys."""
    normalized: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value

    return normalized


_SEQUENTIAL_INSERTS = (
    ".in_layers.",
    ".emb_layers.",
    ".out_layers.",
    ".to_out.",
)


def map_sdxl_to_native(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map a normalized SDXL UNet state dict to our module naming."""
    result: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("time_embed."):
            new_key = "time_embed.layers." + new_key[len("time_embed."):]
        elif new_key.startswith("label_emb.0."):
            # Collapses the redundant outer Sequential wrapper AND inserts
            # ".layers." in one step: "label_emb.0.0.weight" -> "label_emb.layers.0.weight"
            new_key = "label_emb.layers." + new_key[len("label_emb.0."):]
        elif new_key.startswith("out."):
            new_key = "out.layers." + new_key[len("out."):]
        elif ".ff.net." in new_key:
            new_key = new_key.replace(".ff.net.", ".ff.layers.")
        else:
            for token in _SEQUENTIAL_INSERTS:
                if token in new_key:
                    new_key = new_key.replace(token, token[:-1] + ".layers.")
                    break

        result[new_key] = value

    return result
