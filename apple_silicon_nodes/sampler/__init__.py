"""
MLX Native Sampler
==================
Euler sampler running entirely in MLX on Apple Silicon.

Submodules:
  - cache: TeaCacheState (output-level step skipping)
  - core: core denoising loop, sigma scheduling, acceleration helpers

This module provides the ComfyUI node interface (ASDX_MLXSampler)
that wraps the core sampling logic.
"""

from __future__ import annotations

from typing import Any

import torch

from .. import metadata as metadata_util
from . import bridge
from .core import _SamplerCore


# ── Preview cache ─────────────────────────────────────────────────────

_PREVIEW_CACHE: dict[str, Any] = {}


# ── Sampler Node ──────────────────────────────────────────────────────

class ASDX_MLXSampler:
    """MLX-native FLUX sampler with SeaCache acceleration.

    Runs the full denoising loop in MLX, only bridging to PyTorch
    for the final latent output (for ComfyUI compatibility).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("asdx_model",),
                "positive": ("mlx_conditioning",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "teacache": ("BOOLEAN", {"default": False}),
                "teacache_threshold": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 1.0, "step": 0.01}),
                "kontext": ("BOOLEAN", {"default": False}),
                "kontext_reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seacache": ("BOOLEAN", {"default": False}),
                "preview": ("BOOLEAN", {"default": False}),
                "low_memory_mode": ("BOOLEAN", {"default": False}),
                "save_metadata": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Kontext reference — only meaningful when "kontext" above is
                # True; must live here (not "required") so the graph doesn't
                # force a link on every generation, including non-Kontext ones.
                "kontext_reference_latent": ("LATENT", {"default": None}),
                # Mode routing (Phase 2)
                "mode": (["auto", "text2img", "img2img", "inpaint", "fill", "depth"],
                         {"default": "auto"}),
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "image_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64}),
                "mask_padding": ("INT", {"default": 48, "min": 32, "max": 256}),
                "depth_image": ("IMAGE", {"default": None}),
                "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "noise_aug": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Krea2 Identity Edit
                "source_latent": ("LATENT", {"default": None}),
                "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                                          "tooltip": "reference-fidelity dial: multiplies target->source attention. "
                                                     "1.0 = off, >1 pulls harder toward the source's appearance."}),
                "krea2_enhancer_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Krea2-only: community txtfusion conditioning-strength "
                                                     "boost (ported from ComfyUI-Krea2T-Enhancer). 0 = off "
                                                     "(vanilla Krea2, weaker prompt adherence and lower final "
                                                     "latent amplitude); 1.0 matches the reference node's "
                                                     "default (max -- only strength=1.0 has been verified "
                                                     "NaN/overflow-safe against the real node). No effect on "
                                                     "non-Krea2 models."}),
                # Legacy
                "lora_schedule": ("ASDX_LORA_SCHEDULE", {"default": None}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "ASDX/Sampling"

    def sample(
        self,
        model: dict,
        positive: dict,
        latent_image: dict,
        seed: int,
        steps: int,
        guidance: float,
        teacache: bool,
        teacache_threshold: float,
        kontext: bool,
        kontext_reference_strength: float,
        seacache: bool,
        preview: bool,
        low_memory_mode: bool,
        save_metadata: bool,
        kontext_reference_latent: dict | None = None,
        # Mode routing params (Phase 2)
        mode: str = "auto",
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        image_strength: float = 0.8,
        mask_blur: int = 0,
        mask_padding: int = 48,
        depth_image: torch.Tensor | None = None,
        depth_strength: float = 1.0,
        noise_aug: float = 0.0,
        # Krea2 Identity Edit
        source_latent: dict | None = None,
        ref_boost: float = 1.0,
        krea2_enhancer_strength: float = 1.0,
        # Legacy
        lora_schedule: dict | None = None,
    ) -> tuple[dict]:
        """Execute the MLX-native sampling loop via _SamplerCore."""
        transformer = model["transformer"]
        config = model["config"]
        model_type = model.get("model_type", "dev")
        capability = model.get("capability")
        controlnet = model.get("controlnet")

        # Prepare noise. SDXL's UNet consumes the latent grid directly (no
        # 2x2 patchify like FLUX/Krea2), so it needs its own noise-prep path.
        # Flux2/Klein has 128 channels / patch_size=1 / 16x VAE downscale
        # (vs FLUX's 16ch/patch=2/8x) — also its own path. Z-Image shares
        # FLUX's channel count/scale/shift but NOT its patch-token axis
        # order (comfy/ldm/lumina/model.py packs [pH,pW,C], FLUX packs
        # [C,pH,pW] — verified against the real comfy source), so it needs
        # its own noise-prep path too, despite the identical latent shape.
        if model_type == "sdxl":
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_sdxl(
                latent_image, int(seed), config.mlx_dtype
            )
        elif model_type == "flux2":
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_flux2(
                latent_image, int(seed), config.mlx_dtype
            )
        elif model_type in ("zimage", "zimage_turbo"):
            noise, height, width, output_shape = bridge.prepare_noise_from_latent_zimage(
                latent_image, int(seed), config.mlx_dtype
            )
        else:
            noise, height, width, output_shape = bridge.prepare_noise_from_latent(
                latent_image, int(seed), config.mlx_dtype
            )

        # Get previewer for real-time output
        previewer, preview_device = self._get_previewer() if preview else (None, None)

        # Create core sampler with mode routing params
        core = _SamplerCore(
            transformer=transformer,
            config=config,
            positive=positive,
            noise=noise,
            height=height,
            width=width,
            output_shape=output_shape,
            model_type=model_type,
            guidance=guidance,
            teacache=teacache,
            teacache_threshold=teacache_threshold,
            kontext=kontext,
            kontext_reference_latent=kontext_reference_latent,
            kontext_reference_strength=kontext_reference_strength,
            seacache=seacache,
            preview=preview,
            lora_schedule=lora_schedule,
            previewer=previewer,
            preview_device=preview_device,
            capability=capability,
            # Mode routing
            mode=mode,
            # Low memory mode
            low_memory_mode=low_memory_mode,
            image=image,
            image_strength=image_strength,
            mask=mask,
            mask_blur=mask_blur,
            mask_padding=mask_padding,
            depth_image=depth_image,
            depth_strength=depth_strength,
            noise_aug=noise_aug,
            # Krea2 Identity Edit
            source_latent=source_latent,
            ref_boost=ref_boost,
            krea2_enhancer_strength=krea2_enhancer_strength,
            controlnet=controlnet,
        )

        # Run sampling
        out_latent = core.run(steps, seed)

        # Attach generation metadata to the latent output
        if save_metadata:
            out_latent["asdx_metadata"] = metadata_util.build_generation_metadata(
                model_name=model.get("name", "unknown"),
                model_type=model_type,
                precision=config.dtype,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                cfg=guidance,
                mode=mode,
            )

        return (out_latent,)

    @staticmethod
    def _get_previewer():
        """Get latent previewer from ComfyUI."""
        try:
            import comfy.model_management
            import latent_preview
            from comfy.cli_args import LatentPreviewMethod

            device = comfy.model_management.get_torch_device()
            preview_method = getattr(latent_preview, 'args', None)
            if preview_method is not None:
                preview_method = preview_method.preview_method

            cache_key = f"preview:{preview_method}:{device}"
            if cache_key in _PREVIEW_CACHE:
                return _PREVIEW_CACHE[cache_key]

            import comfy.latent_formats
            latent_format = comfy.latent_formats.Flux()
            if preview_method == LatentPreviewMethod.NoPreviews:
                _PREVIEW_CACHE[cache_key] = (None, None)
                return _PREVIEW_CACHE[cache_key]

            previewer = latent_preview.get_previewer(device, latent_format)
            _PREVIEW_CACHE[cache_key] = (previewer, device)
            return _PREVIEW_CACHE[cache_key]
        except Exception:
            return (None, None)


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ASDX_MLXSampler": ASDX_MLXSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_MLXSampler": "🍏 ASDX MLX Native Sampler",
}
