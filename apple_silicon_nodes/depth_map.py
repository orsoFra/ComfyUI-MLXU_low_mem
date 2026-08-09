"""Depth map generation using MLX-compatible depth estimation.

Mirrors the MfluxDepthMap node from ComfyUI-mflux-AnyModel: generates
depth maps from RGB images using a pre-trained depth model. The depth
output is returned as a MFLUX_IMAGE for chaining into depth-controlled
sampling workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)

# ── Cache ──────────────────────────────────────────────────────────────

_DEPTH_CACHE: dict[str, Any] = {}


# ── Node ───────────────────────────────────────────────────────────────

class ASDX_DepthMap(io.ComfyNode):
    """Generate a depth map from an RGB image.

    Uses a pre-trained depth estimation model (DepthPro or similar)
    running on MPS/CPU. Returns a MFLUX_IMAGE with the depth map field
    populated for chaining into depth-controlled sampling.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_DepthMap",
            display_name="🍏 ASDX Depth Map",
            category="ASDX/Depth",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("resolution", default=1024, min=256, max=2048, step=64, optional=True),
            ],
            outputs=[
                io.Custom("mflux_image").Output(display_name="depth_image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        resolution: int = 1024,
    ) -> io.NodeOutput:
        """Generate depth map from image.

        Returns a MFLUX_IMAGE with the depth map in the depth_map field.
        """
        from .image_chain import MFLUX_IMAGE

        try:
            depth_array = cls._run_depth_model(image, resolution)
            return io.NodeOutput(MFLUX_IMAGE(
                image=image,
                depth_map=depth_array,
                source="depth",
                metadata={"resolution": resolution},
            ))
        except Exception as e:
            logger.error("ASDX_DepthMap failed: %s", e)
            # Fallback: return image-only payload
            return io.NodeOutput(MFLUX_IMAGE(image=image, source="depth_error"))

    @classmethod
    def _run_depth_model(cls, image: torch.Tensor, resolution: int) -> Any:
        """Run depth estimation on the image.

        Tries multiple backends in order:
        1. transformers (DepthPro)
        2. Simple monocular depth approximation (fallback)

        Returns an MLX array or numpy array.
        """
        # Try transformers + DepthPro first
        try:
            return cls._depth_pro_depth(image, resolution)
        except ImportError:
            logger.debug("DepthPro not available, using fallback")
        except Exception as e:
            logger.debug("DepthPro failed: %s, using fallback", e)

        # Fallback: compute a simple depth approximation from luminance
        return cls._fallback_depth(image, resolution)

    @staticmethod
    def _depth_pro_depth(image: torch.Tensor, resolution: int) -> Any:
        """Run DepthPro via transformers library."""
        import mlx.core as mx
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor

        # Resize image to target resolution
        orig_h, orig_w = image.shape[1], image.shape[2]
        scale = resolution / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        # Convert to PIL
        pil_image = ASDX_DepthMap._tensor_to_pil(image)

        # Load model (cached)
        cache_key = f"depth_pro:{resolution}"
        if cache_key not in _DEPTH_CACHE:
            # Only one depth model is meaningfully "current" at a time -- the
            # `resolution` widget alone (256-2048, step 64) can produce up to
            # 28 distinct keys, each holding a full DepthPro model on MPS.
            # Evict prior entries before loading a new one instead of
            # accumulating one per resolution ever used in the session (same
            # fix already applied to loader.py's _MODEL_CACHE).
            if _DEPTH_CACHE:
                from . import bridge
                _DEPTH_CACHE.clear()
                bridge.clear_mlx_cache()

            processor = AutoImageProcessor.from_pretrained(
                "facebook/depth-pro-foundation",
                trust_remote_code=True,
            )
            model = AutoModelForDepthEstimation.from_pretrained(
                "facebook/depth-pro-foundation",
                trust_remote_code=True,
            )
            if torch.backends.mps.is_available():
                model = model.to("mps")
            model.eval()
            _DEPTH_CACHE[cache_key] = (processor, model)

        processor, model = _DEPTH_CACHE[cache_key]

        # Process image
        inputs = processor(images=pil_image, return_tensors="pt")
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract depth map
        predicted_depth = outputs.predicted_depth

        # Resize to original resolution
        import torch.nn.functional as F
        depth_resized = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).squeeze(0)

        # Normalize to [0, 1]
        depth_min = depth_resized.min()
        depth_max = depth_resized.max()
        if depth_max > depth_min:
            depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = depth_resized

        # Convert to numpy
        depth_np = depth_normalized.cpu().numpy().astype(np.float32)

        # Convert to MLX array
        return mx.array(depth_np)

    @staticmethod
    def _fallback_depth(image: torch.Tensor, resolution: int) -> Any:
        """Generate a simple depth approximation from luminance.

        This is a rough approximation: darker areas = farther, lighter = closer.
        Not a real depth model, but provides a usable depth-like signal.
        """
        import mlx.core as mx

        # Convert to grayscale (luminance)
        if image.ndim == 4:
            # [B, H, W, C]
            gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        else:
            gray = image

        # Invert: darker = farther (higher depth value)
        depth = 1.0 - gray

        # Convert to MLX array
        depth_np = depth.numpy().astype(np.float32) if hasattr(depth, "numpy") else np.array(depth, dtype=np.float32)
        return mx.array(depth_np)

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor):
        """Convert a torch tensor to PIL Image."""
        from PIL import Image

        # Handle batch dimension
        if image.ndim == 4:
            image = image[0]  # Take first image

        # [H, W, C] -> [C, H, W] -> PIL
        if image.ndim == 3:
            image = image.permute(2, 0, 1)

        # Normalize to [0, 255] uint8
        img_np = image.cpu().numpy().astype(np.float32)
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(np.transpose(img_np, (1, 2, 0)))


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_LIST = [
    ASDX_DepthMap,
]
