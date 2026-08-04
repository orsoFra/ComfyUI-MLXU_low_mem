"""
ControlNet Union Support
========================
MLX-native ControlNet Union for FLUX.1, matching the reference architecture
in `comfy/ldm/flux/controlnet.py` (ControlNetFlux): a FLUX transformer that
shares img_in/txt_in/time_in/double_blocks/single_blocks with the base model,
producing per-block residuals injected into the base model's forward pass.

Supports up to 8 control types (Union checkpoints):
  pose, depth, soft_edge, line_canny, normal, segment, tile, repaint

Submodules:
  - types: CONTROL_NET_TYPES constant
  - model: ControlNetFlux, load_controlnet_union
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from .types import CONTROL_NET_TYPES
from .model import ControlNetFlux, load_controlnet_union


# ── Nodes ─────────────────────────────────────────────────────────────

class ASDX_ControlNetUnionLoader:
    """Load a ControlNet Union model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_name": (cls._get_controlnets(),),
            },
        }

    RETURN_TYPES = ("controlnet",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load"
    CATEGORY = "ASDX/ControlNet"

    @staticmethod
    def _get_controlnets() -> list[str]:
        """Get list of available ControlNet models."""
        try:
            import folder_paths
            cns = []
            for folder in ("controlnet",):
                try:
                    cns.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if cns:
                return cns
        except Exception:
            pass
        return ["controlnet_union.safetensors"]

    def load(self, control_net_name: str) -> tuple[dict]:
        path = self._resolve_path(control_net_name)
        control_net = load_controlnet_union(path)
        return ({"control_net": control_net, "name": control_net_name},)

    @staticmethod
    def _resolve_path(name: str) -> Path:
        try:
            import folder_paths
            for folder in ("controlnet",):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "controlnet" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


class ASDX_ApplyControlNet:
    """Apply ControlNet conditioning to a diffusion model.

    VAE-encodes the control image into a latent, matching the reference
    ControlNet forward pass (`ControlNetFlux` consumes a VAE-encoded,
    latent-format-processed control latent — not raw pixels). The encoded
    latent, packed the same way as the noise, is stored on the model dict;
    the sampler computes per-step residuals and injects them into every
    FLUX double/single block.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("asdx_model",),
                "control_net": ("controlnet",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "control_type": (["pose", "depth", "soft_edge", "line_canny", "normal", "segment", "tile", "repaint"],),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
            },
        }

    RETURN_TYPES = ("asdx_model",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "ASDX/ControlNet"

    def apply(
        self,
        model: dict,
        control_net: dict,
        image: torch.Tensor,
        vae: Any,
        strength: float,
        control_type: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[dict]:
        """Attach ControlNet conditioning to the model."""
        type_idx = CONTROL_NET_TYPES.get(control_type, 0)

        control_image = image if mask is None else self._concat_mask(image, mask)

        model = dict(model)
        model["controlnet"] = {
            "control_net": control_net["control_net"],
            "image": control_image,
            "vae": vae,
            "strength": strength,
            "control_type": [type_idx],
        }

        print(f"[ASDX] ControlNet applied: {control_type} strength={strength:.2f}")
        return (model,)

    @staticmethod
    def _concat_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Concatenate a mask as an extra image channel (inpaint ControlNet-Union)."""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(-1).to(image.dtype)
        return torch.cat([image, mask], dim=-1)


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ASDX_ControlNetUnionLoader": ASDX_ControlNetUnionLoader,
    "ASDX_ApplyControlNet": ASDX_ApplyControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_ControlNetUnionLoader": "🍏 ASDX ControlNet Union Loader",
    "ASDX_ApplyControlNet": "🍏 ASDX Apply ControlNet",
}
