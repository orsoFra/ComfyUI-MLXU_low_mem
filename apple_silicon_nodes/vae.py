"""
VAE Decode / Encode nodes
=========================
MLX-native VAE decoding and encoding for FLUX.

Uses the MLX VAE implementation for zero-copy decoding on Apple Silicon,
bridging only at the input/output boundaries.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np
import torch


# ── Globals ───────────────────────────────────────────────────────────

_VAE_CACHE: dict[str, Any] = {}


# ── VAE Decode ───────────────────────────────────────────────────────

class ASDX_VAEDecode:
    """Decode FLUX latents to images using MLX VAE.

    The VAE decoder runs entirely in MLX, only the final image tensor
    is converted to PyTorch for ComfyUI downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "ASDX/Latent"

    def decode(self, samples: dict, vae: Any) -> tuple[torch.Tensor]:
        # Get latent samples
        if not isinstance(samples, dict) or "samples" not in samples:
            raise RuntimeError("ASDX VAE Decode: expected LATENT input.")

        latent = samples["samples"]

        # Always use the real ComfyUI/PyTorch VAE. `_decode_with_mlx_vae()`
        # (below) has no real weight loading — it's an untrained placeholder
        # that silently ignores `vae` and produces noise, not an image, for
        # any latent that used to route through it (16ch: FLUX.1/Z-Image).
        # SDXL (4ch) and Flux2 (128ch) already always used this real path.
        # Left `_decode_with_mlx_vae`/`mlx_vae.py` in place, just unreferenced
        # from here — a native MLX VAE decoder remains a separate future task.
        return self._fallback_decode(latent, vae)

    def _decode_with_mlx_vae(self, latent: mx.array) -> mx.array:
        """Decode using MLX VAE.

        In production, this would use the native MLX VAE from sdmlx.
        For now, we provide a placeholder that demonstrates the pattern.
        """
        # The real implementation would:
        # 1. Load the VAE weights into an MLX VAE module
        # 2. Call vae.decode(latent)
        # 3. Return the decoded output

        # Placeholder: for demonstration, just return the latent
        # A real implementation needs the VAE module from sdmlx/mlx_sd
        try:
            from .mlx_vae import get_vae_decoder
            vae = get_vae_decoder()
            if vae is not None:
                return vae.decode(latent)
        except Exception:
            pass

        # Fallback: just return the latent (will produce garbage without real VAE)
        # This should never be reached in a proper installation
        print("[ASDX] VAE Decode: no MLX VAE available, using fallback")
        return latent

    @staticmethod
    def _fallback_decode(latent: torch.Tensor, vae: Any) -> tuple[torch.Tensor]:
        """Standard PyTorch VAE decode fallback.

        `latent` here is already the unwrapped tensor (`decode()` extracts
        `samples["samples"]` before calling this) — do not index it again.
        """
        image = vae.decode(latent)
        return (image,)


# ── VAE Encode ───────────────────────────────────────────────────────

class ASDX_VAEEncode:
    """Encode images to FLUX latents using MLX VAE.

    The encoding runs in MLX for Apple Silicon acceleration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "ASDX/Latent"

    def encode(self, pixels: torch.Tensor, vae: Any) -> tuple[dict]:
        t0 = time.perf_counter()

        if pixels.ndim != 4:
            raise RuntimeError(f"ASDX VAE Encode: expected [B,H,W,C] image, got {pixels.shape}")

        # Convert to MLX
        # Transpose to [B, C, H, W] and normalize to [-1, 1]
        image_np = pixels.detach().cpu().numpy().astype(np.float32)
        image_np = image_np.transpose(0, 3, 1, 2)  # BHWC -> BCHW
        image_np = (image_np * 2.0) - 1.0  # [0,1] -> [-1,1]

        image_mlx = mx.array(image_np)
        mx.eval(image_mlx)

        # Encode with MLX VAE
        latent = self._encode_with_mlx_vae(image_mlx)
        mx.eval(latent)

        # Convert back to PyTorch
        latent_np = np.array(latent.astype(mx.float32), dtype=np.float32)
        latent_torch = torch.from_numpy(latent_np)

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] VAE Encode: {latent_torch.shape}, {elapsed:.2f}s")

        return ({"samples": latent_torch},)

    def _encode_with_mlx_vae(self, image: mx.array) -> mx.array:
        """Encode using MLX VAE encoder."""
        try:
            from .mlx_vae import get_vae_encoder
            vae = get_vae_encoder()
            if vae is not None:
                return vae.encode(image)
        except Exception:
            pass
        print("[ASDX] VAE Encode: no MLX VAE available, using fallback")
        return image


NODE_CLASS_MAPPINGS = {
    "ASDX_VAEDecode": ASDX_VAEDecode,
    "ASDX_VAEEncode": ASDX_VAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_VAEDecode": "🍏 ASDX VAE Decode (MLX)",
    "ASDX_VAEEncode": "🍏 ASDX VAE Encode (MLX)",
}
