"""
State dict weight mapping for Krea2 checkpoints.

Krea2 uses a different key naming convention from FLUX.1:
- `first.weight` instead of `img_in.weight`
- `blocks.{i}.attn.wq/wk/wv/gate/wo` for attention
- `blocks.{i}.mod.lin` for DoubleSharedModulation (6 params)
- `blocks.{i}.prenorm.scale` / `blocks.{i}.postnorm.scale` for RMSNorm
- `blocks.{i}.attn.qknorm.qnorm.scale` / `knorm.scale` for QK norm
- `txtfusion.*` for text fusion adapter
- `txtmlp.*` for text MLP (RMSNorm + 2 Linear layers)
- `tmlp.0/2` and `tproj.1` for time embedding
- `last.linear.*` for output projection
- `last.modulation.lin` for SimpleModulation
- `last.norm.scale` for LastLayer RMSNorm

This module handles:
1. Prefix normalization (strip diffusion_model. etc.)
2. Key mapping from Krea2 format to native naming
3. txtmlp/tproj/tmlp index mapping (checkpoint: module.{i}.* -> MLX Sequential's module.layers.{i}.*)

txtfusion.projector.weight needs NO transpose: nn.Linear(12, 1).weight has
shape (out_features, in_features) = (1, 12) in MLX, which already matches
the checkpoint's stored [1, 12] shape directly.
"""

from __future__ import annotations

import mlx.core as mx


def normalize_krea2_keys(
    state_dict: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Normalize Krea2 checkpoint keys by stripping common prefixes.

    Args:
        state_dict: Raw state dict from checkpoint file.

    Returns:
        State dict with cleaned keys.
    """
    normalized: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value

    return normalized


def map_krea2_to_native(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map Krea2 checkpoint keys to native naming convention.

    Key mappings:
        first.weight                    → first.weight (keep)
        blocks.{i}.attn.*               → blocks.{i}.attn.* (keep, gate→gate_proj)
        blocks.{i}.mod.lin              → blocks.{i}.mod.lin (keep)
        blocks.{i}.prenorm/postnorm     → blocks.{i}.prenorm/postnorm (keep)
        blocks.{i}.attn.qknorm.*        → blocks.{i}.attn.qknorm.* (keep)
        tmlp.{i}.*                      → tmlp.layers.{i}.* (insert .layers)
        tproj.{i}.*                     → tproj.layers.{i}.* (insert .layers)
        txtmlp.{i}.*                    → txtmlp.layers.{i}.* (insert .layers)
        txtfusion.projector.weight      → txtfusion.projector.weight (transpose)
        txtfusion.*.norm.scale          → txtfusion.*.norm.weight
        last.*                          → last.* (keep)

    The .layers insertion handles MLX Sequential parameter naming:
    MLX stores Sequential params as module.layers.{index}.{param}
    while the checkpoint uses flat indices module.{index}.{param}.

    Args:
        state_dict: State dict from a Krea2 checkpoint.

    Returns:
        State dict with keys mapped to native convention.
    """
    result: dict[str, mx.array] = {}

    for key, value in state_dict.items():
        new_key = key

        # ── blocks.{i}.attn.gate → blocks.{i}.attn.gate_proj ───────
        #    txtfusion.*.attn.gate → txtfusion.*.attn.gate_proj ───────
        if ".attn.gate.weight" in new_key:
            new_key = new_key.replace(".attn.gate.", ".attn.gate_proj.")

        # ── txtfusion.norm.scale → norm.weight ─────────────────────
        elif "txtfusion." in new_key and ".norm.scale" in new_key:
            new_key = new_key.replace(".norm.scale", ".norm.weight")

        # ── Insert .layers for Sequential submodules ───────────────
        # MLX Sequential stores params as module.layers.{idx}.{param}
        # Checkpoint uses flat: module.{idx}.{param}
        elif new_key.startswith(("tmlp.", "tproj.", "txtmlp.")):
            # Split: "txtmlp.1.weight" → ["txtmlp", "1", "weight"]
            parts = new_key.split(".", 2)
            if len(parts) == 3:
                new_key = f"{parts[0]}.layers.{parts[1]}.{parts[2]}"
            else:
                new_key = f"{new_key}.layers"

        # ── All other keys: keep as-is ─────────────────────────────
        result[new_key] = value
        continue

    return result
