"""
Empty FLUX Latent
=================
Creates empty 16-channel FLUX latents optimized for Apple Silicon.
"""

from __future__ import annotations

from typing import Any

import torch

import comfy.model_management


# ── Resolution presets ───────────────────────────────────────────────

# FLUX works best with resolutions divisible by 16.
# These are common aspect ratios optimized for Apple Silicon memory.
_RESOLUTION_PRESETS = {
    "1024 x 1024 (1:1)": (1024, 1024),
    "1024 x 1024 (1:1) XL": (1152, 1152),
    "832 x 1216 (5:8)": (832, 1216),
    "1216 x 832 (8:5)": (1216, 832),
    "896 x 1152 (3:4)": (896, 1152),
    "1152 x 896 (4:3)": (1152, 896),
    "768 x 1344 (9:16)": (768, 1344),
    "1344 x 768 (16:9)": (1344, 768),
    "640 x 1344 (2:5)": (640, 1344),
    "1344 x 640 (5:2)": (1344, 640),
}


class ASDX_EmptyFLUXLatent:
    """Create an empty 16-channel FLUX latent tensor.

    The latent is placed on the appropriate device (MPS when available)
    for zero-copy with the MLX sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "aspect_ratio": (["auto"] + list(_RESOLUTION_PRESETS.keys()), {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "ASDX/Latent"

    def generate(
        self,
        width: int,
        height: int,
        batch_size: int,
        aspect_ratio: str = "auto",
    ) -> tuple[dict]:
        # Apply aspect ratio preset if selected
        if aspect_ratio != "auto" and aspect_ratio in _RESOLUTION_PRESETS:
            width, height = _RESOLUTION_PRESETS[aspect_ratio]

        # Ensure divisible by 16
        width = (width // 16) * 16
        height = (height // 16) * 16

        # Create latent on MPS device for zero-copy with sampler
        device = self._get_device()
        latent = torch.zeros(
            [batch_size, 16, height // 8, width // 8],
            device=device,
            dtype=comfy.model_management.intermediate_dtype(),
        )

        print(f"[ASDX] Empty FLUX Latent: {width}x{height}, batch={batch_size}, "
              f"latent_shape=[{batch_size}, 16, {height//8}, {width//8}]")

        return ({"samples": latent, "downscale_ratio_spacial": 8},)

    @staticmethod
    def _get_device() -> torch.device:
        """Get the best available device."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")


NODE_CLASS_MAPPINGS = {
    "ASDX_EmptyFLUXLatent": ASDX_EmptyFLUXLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_EmptyFLUXLatent": "🍏 ASDX Empty FLUX Latent",
}
