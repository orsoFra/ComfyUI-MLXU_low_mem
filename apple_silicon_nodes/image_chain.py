"""MFLUX_IMAGE typed chain for multi-image workflows.

Mirrors the MfluxImage pattern from ComfyUI-mflux-AnyModel: a typed
container that passes between nodes for img2img, inpainting, depth
control, and multi-image edit/redux workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np
import torch


# ── MFLUX_IMAGE dataclass ─────────────────────────────────────────────

@dataclass
class MFLUX_IMAGE:
    """Typed image payload for chaining in multi-image workflows.

    Combines image, latent, depth map, and mask into a single typed
    container that can be passed between nodes without serialization.
    """

    image: torch.Tensor | None = None
    """[B, H, W, C] float32 [0, 1] — the original or generated image."""

    latent: torch.Tensor | None = None
    """[B, 16, H/8, W/8] — pre-encoded latent (FLUX format)."""

    depth_map: mx.array | None = None
    """[B, H/8, W/8, 1] — depth map from DepthPro (MLX array)."""

    mask: torch.Tensor | None = None
    """[B, H, W] float32 [0, 1] — inpainting mask."""

    source: str = "input"
    """Source identifier: 'input', 'generated', 'reference', 'depth'."""

    metadata: dict = field(default_factory=dict)
    """Arbitrary metadata (strength, resolution, etc.)."""

    @property
    def has_image(self) -> bool:
        return self.image is not None and self.image.numel() > 0

    @property
    def has_latent(self) -> bool:
        return self.latent is not None and self.latent.numel() > 0

    @property
    def has_mask(self) -> bool:
        return self.mask is not None and self.mask.numel() > 0

    @property
    def has_depth(self) -> bool:
        return self.depth_map is not None and self.depth_map.size > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ComfyUI bridge (type hints only, no tensor data)."""
        return {
            "type": "mflux_image",
            "has_image": self.has_image,
            "has_latent": self.has_latent,
            "has_mask": self.has_mask,
            "has_depth": self.has_depth,
            "source": self.source,
            "metadata": self.metadata,
        }


# ── Node: ImageToLatent ───────────────────────────────────────────────

class ASDX_ImageToLatent:
    """VAE-encode an image to latent, return MFLUX_IMAGE.

    Encodes a [B, H, W, C] image tensor to a [B, 16, H/8, W/8] FLUX
    latent using the MLX VAE encoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("mflux_image",)
    RETURN_NAMES = ("image_latent",)
    FUNCTION = "encode"
    CATEGORY = "ASDX/ImageChain"

    def encode(self, image: torch.Tensor) -> tuple[MFLUX_IMAGE]:
        try:
            from . import bridge as mlx_bridge
            from .mlx_vae import MLXVAE

            # Transpose to [B, C, H, W] for VAE
            img_bw = image.permute(0, 3, 1, 2) if image.ndim == 4 else image
            # Normalize from [0, 1] to [-1, 1] for VAE
            img_normalized = (img_bw - 0.5) / 0.5

            # Use MLX VAE encoder
            vae = MLXVAE()
            latent = vae.encode(img_normalized)

            # Convert back to PyTorch
            latent_pt = torch.from_numpy(
                np.array(latent.cpu().numpy(), dtype=np.float32)
            )

            return (MFLUX_IMAGE(
                image=image,
                latent=latent_pt,
                source="input",
                metadata={"width": image.shape[2], "height": image.shape[1]},
            ),)
        except Exception as e:
            print(f"[ASDX_ImageToLatent] Error: {e}")
            # Fallback: return image-only payload
            return (MFLUX_IMAGE(image=image, source="input"),)


# ── Node: MaskFromImage ───────────────────────────────────────────────

class ASDX_MaskFromImage:
    """Generate a binary mask from an image using threshold.

    Converts a grayscale or RGB image to a binary [0, 1] mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "mflux_image",)
    RETURN_NAMES = ("mask", "image_with_mask",)
    FUNCTION = "create"
    CATEGORY = "ASDX/ImageChain"

    def create(
        self,
        image: torch.Tensor,
        threshold: float,
        invert: bool,
    ) -> tuple[torch.Tensor, MFLUX_IMAGE]:
        # Convert to grayscale if needed
        if image.ndim == 4 and image.shape[-1] > 1:
            mask = image.mean(dim=-1, keepdim=True).squeeze(-1)
        else:
            mask = image.squeeze(-1) if image.ndim == 4 else image

        # Threshold
        mask = (mask > threshold).float()
        if invert:
            mask = 1.0 - mask

        # Expand to [B, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        payload = MFLUX_IMAGE(
            image=image,
            mask=mask,
            source="mask",
            metadata={"threshold": threshold, "invert": invert},
        )
        return (mask, payload)


# ── Node: MaskBlur ────────────────────────────────────────────────────

class ASDX_MaskBlur:
    """Apply Gaussian blur to a mask for soft edges."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("INT", {"default": 4, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("MASK", "mflux_image",)
    RETURN_NAMES = ("blurred_mask", "image_with_mask",)
    FUNCTION = "blur"
    CATEGORY = "ASDX/ImageChain"

    def blur(
        self,
        mask: torch.Tensor,
        blur_radius: int,
    ) -> tuple[torch.Tensor, MFLUX_IMAGE]:
        if blur_radius <= 0:
            return (mask, MFLUX_IMAGE(mask=mask, source="mask"))

        try:
            import torch.nn.functional as F

            # Pad mask for edge convolution
            pad = blur_radius // 2
            if pad > 0:
                padded = F.pad(mask, (pad, pad, pad, pad), mode="reflect")
                # 2D Gaussian blur via separable convolution
                k_size = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                kernel = self._gaussian_kernel(k_size, blur_radius / 3.0)
                blurred = F.conv2d(
                    padded.unsqueeze(1), kernel, padding=k_size // 2
                ).squeeze(1)
                # Crop to original size
                if pad > 0:
                    blurred = blurred[:, pad:-pad, pad:-pad]
            else:
                blurred = mask
        except Exception:
            blurred = mask

        if blurred.ndim == 3 and blurred.shape[0] > 1:
            pass  # batch already correct
        elif blurred.ndim == 2:
            blurred = blurred.unsqueeze(0)

        return (blurred, MFLUX_IMAGE(mask=blurred, source="mask_blur"))

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 1D Gaussian kernel, then make it 2D via outer product."""
        import math
        coords = torch.arange(kernel_size, dtype=torch.float32)
        center = kernel_size // 2
        diff = (coords - center).float() / sigma
        kernel_1d = torch.exp(-0.5 * diff * diff)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d.unsqueeze(0).unsqueeze(0)


# ── Node: ImageCompositor ─────────────────────────────────────────────

class ASDX_ImageCompositor:
    """Composite a generated image over an original using a mask.

    Implements mask-preserve compositing: where mask=1, keep the original;
    where mask=0, use the generated image. Blended proportionally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "mflux_image",)
    RETURN_NAMES = ("composited", "image_chain",)
    FUNCTION = "composite"
    CATEGORY = "ASDX/ImageChain"

    def composite(
        self,
        original_image: torch.Tensor,
        generated_image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, MFLUX_IMAGE]:
        # Expand mask to match image dimensions
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1).expand(-1, -1, original_image.shape[-1])
        elif mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, original_image.shape[-1])

        # Ensure same size
        h, w = original_image.shape[1], original_image.shape[2]
        if generated_image.shape[1] != h or generated_image.shape[2] != w:
            generated_image = torch.nn.functional.interpolate(
                generated_image.permute(0, 3, 1, 2),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        # Clamp mask to [0, 1]
        mask = mask.clamp(0.0, 1.0)

        # Composite: original * mask + generated * (1 - mask)
        composited = original_image * mask + generated_image * (1.0 - mask)
        composited = composited.clamp(0.0, 1.0)

        payload = MFLUX_IMAGE(
            image=composited,
            source="composited",
            metadata={"mask_used": True},
        )
        return (composited, payload)
