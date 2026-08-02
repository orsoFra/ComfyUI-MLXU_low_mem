"""
ControlNet Union Support
========================
MLX-native ControlNet Union ProMax for FLUX.

Supports 8 control types:
  pose, depth, soft_edge, line_canny, normal, segment, tile, repaint

Submodules:
  - types: CONTROL_NET_TYPES constant
  - blocks: ControlNet building blocks (CondEmbedding, TimeEmbedding, SinusoidalPositionalEncoding)
  - model: ControlNetUnionModel, load_controlnet_union, weight assignment
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from .. import bridge
from .types import CONTROL_NET_TYPES
from .model import ControlNetUnionModel, load_controlnet_union


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

    Injects ControlNet residuals into the transformer's attention layers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("asdx_model",),
                "control_net": ("controlnet",),
                "image": ("IMAGE",),
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
        strength: float,
        control_type: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[dict]:
        """Attach ControlNet conditioning to the model."""
        type_idx = CONTROL_NET_TYPES.get(control_type, 0)

        # Prepare control image: [B, H, W, 3] -> [B, 4, H, W]
        control_img = self._prepare_control_image(image, mask)

        # Store ControlNet info on the model
        model["controlnet"] = {
            "control_net": control_net["control_net"],
            "image": control_img,
            "strength": strength,
            "control_type": type_idx,
        }

        print(f"[ASDX] ControlNet applied: {control_type} strength={strength:.2f}")
        return (model,)

    @staticmethod
    def _prepare_control_image(image: torch.Tensor, mask: torch.Tensor | None) -> mx.array:
        """Convert image (+ optional mask) to MLX array [B, 4, H, W]."""
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Normalize to [-1, 1]
        img_np = image.detach().cpu().numpy().astype(np.float32)
        img_np = (img_np * 2.0) - 1.0

        # Add mask channel if provided
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            # Transpose mask to [B, 1, H, W]
            if mask_np.ndim == 3:
                mask_np = mask_np[:, None, :, :]
            control = np.concatenate([img_np, mask_np], axis=1)
        else:
            # 3 channels RGB -> add zeros for alpha
            control = np.concatenate([img_np, np.zeros_like(img_np[:, :1])], axis=1)

        control_mlx = mx.array(control)
        mx.eval(control_mlx)
        return control_mlx


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ASDX_ControlNetUnionLoader": ASDX_ControlNetUnionLoader,
    "ASDX_ApplyControlNet": ASDX_ApplyControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_ControlNetUnionLoader": "🍏 ASDX ControlNet Union Loader",
    "ASDX_ApplyControlNet": "🍏 ASDX Apply ControlNet",
}
