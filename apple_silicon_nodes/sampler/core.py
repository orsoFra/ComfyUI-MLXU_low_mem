"""Core sampling logic for the MLX-native FLUX sampler.

Contains the denoising loop, sigma scheduling, and acceleration helpers.
The ComfyUI node (sampler.py) wraps this to provide the node interface.
"""

from __future__ import annotations

import enum
import math
import time
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from .. import capability as cap_module
from . import bridge
from .cache import KontextCache, TeaCacheState


class SamplerMode(enum.Enum):
    """Sampling mode: text2img, img2img, inpainting, fill, depth control."""

    TEXT_TO_IMAGE = "text2img"
    IMAGE_TO_IMAGE = "img2img"
    INPAINTING = "inpaint"
    FILL = "fill"
    DEPTH_CONTROL = "depth"


class _SamplerCore:
    """Core denoising loop and acceleration logic.

    This class contains the sampling logic without ComfyUI node boilerplate.
    It is instantiated by ASDX_MLXSampler for each sample call.
    """

    def __init__(
        self,
        transformer: Any,
        config: Any,
        positive: dict,
        noise: mx.array,
        height: int,
        width: int,
        output_shape: tuple[int, int],
        rope: mx.array,
        txt_projected: mx.array,
        model_type: str,
        guidance: float,
        teacache: bool,
        teacache_threshold: float,
        kontext: bool,
        kontext_reference_latent: dict | None,
        kontext_reference_strength: float,
        seacache: bool,
        preview: bool,
        lora_schedule: dict | None,
        previewer: Any | None,
        preview_device: str | None,
        capability: Any | None = None,
        # Mode routing (Phase 2)
        mode: str = "auto",
        image: torch.Tensor | None = None,
        image_strength: float = 0.8,
        mask: torch.Tensor | None = None,
        mask_blur: int = 0,
        mask_padding: int = 48,
        depth_image: torch.Tensor | None = None,
        depth_strength: float = 1.0,
        noise_aug: float = 0.0,
    ):
        self.transformer = transformer
        self.config = config
        self.positive = positive
        self.noise = noise
        self.height = height
        self.width = width
        self.output_shape = output_shape
        self.model_type = model_type
        self.guidance = guidance
        self.teacache = teacache
        self.teacache_threshold = teacache_threshold
        self.kontext = kontext
        self.kontext_reference_latent = kontext_reference_latent
        self.kontext_reference_strength = kontext_reference_strength
        self.seacache = seacache
        self.preview = preview
        self.lora_schedule = lora_schedule
        self.previewer = previewer
        self.preview_device = preview_device
        self.capability = capability
        # Mode routing
        self.mode = mode
        self.image = image
        self.image_strength = image_strength
        self.mask = mask
        self.mask_blur = mask_blur
        self.mask_padding = mask_padding
        self.depth_image = depth_image
        self.depth_strength = depth_strength
        self.noise_aug = noise_aug

    def run(self, steps: int, seed: int) -> dict:
        """Execute the MLX-native sampling loop and return the result latent."""
        precision = self.config.mlx_dtype
        model_type = self.model_type

        # Detect mode and prepare noise (Phase 2: mode routing)
        self._mode = self._detect_mode()
        if self._mode != SamplerMode.TEXT_TO_IMAGE:
            print(f"[ASDX] Mode: {self._mode.value}")
            if self._mode == SamplerMode.IMAGE_TO_IMAGE:
                self.noise = self._prepare_img2img_noise()
            elif self._mode in (SamplerMode.INPAINTING, SamplerMode.FILL):
                self.noise = self._prepare_inpainting_noise()
            elif self._mode == SamplerMode.DEPTH_CONTROL:
                self.noise = self._prepare_depth_noise()

        # Prepare conditioning
        prompt_embeds, pooled_embeds, cond_guidance = bridge.conditioning_to_mlx(
            self.positive, precision
        )
        effective_guidance = float(self.guidance) if self.guidance > 0 else (
            cond_guidance if cond_guidance is not None else 3.5
        )

        # Capability-aware parameter filtering (mflux-AnyModel pattern)
        if self.capability is not None:
            candidate_params = {
                "guidance": self.guidance,
                "width": self.width,
                "height": self.height,
                "steps": steps,
            }
            try:
                valid_params, dropped = cap_module.filter_params_for_model(
                    self.capability, candidate_params
                )
                if dropped:
                    print(f"[ASDX] Dropped params for {self.capability.name}: {dropped}")
                # Update effective_guidance from filtered params if present
                if "guidance" in valid_params and valid_params["guidance"] is not None:
                    effective_guidance = float(valid_params["guidance"])
            except ValueError as e:
                print(f"[ASDX] Capability filter warning: {e}")

        # Precompute rope embeddings
        rope = self.transformer.get_rope(self.noise.shape[1], prompt_embeds.shape[1])

        # Precompute text projection through txt_in linear layer
        txt_projected = self.transformer.txt_in(prompt_embeds)

        # Setup sigmas for the model type
        sigmas = self._create_sigmas(steps, model_type, self.width, self.height)

        # Setup TeaCache
        teacache_state: TeaCacheState | None = None
        if self.teacache:
            teacache_state = TeaCacheState(
                threshold=self.teacache_threshold,
                warmup_steps=5,
                final_steps=3,
            )

        # Setup Kontext KV cache
        kontext_cache = KontextCache()
        if self.kontext and self.kontext_reference_latent is not None:
            kontext_cache.set(True, reference_tokens=0)
            self._prepare_kontext_reference(kontext_cache, self.kontext_reference_latent,
                                            self.transformer, precision)

        # SeaCache state
        seacache_enabled = self.seacache and model_type == "dev"
        previous_output: mx.array | None = None
        cache_threshold = 0.1 if seacache_enabled else 100.0
        warmup_steps = 3

        # Memory profiling
        mx.reset_peak_memory()
        step_times: list[float] = []

        t_sampling_start = time.perf_counter()
        kontext_ref_tokens = 0
        kontext_applied = False

        for t in range(steps):
            step_start = time.perf_counter()

            # Current sigma
            sigma_t = sigmas[t]
            sigma_next = sigmas[t + 1] if t + 1 < len(sigmas) else 0.0

            # Update LoRA schedule
            if self.lora_schedule is not None:
                self.lora_schedule["step"] = t
                self._update_lora_schedule(self.lora_schedule, t, steps)

            # --- Compute transformer output ---
            if teacache_state is not None:
                current_output = self.transformer.predict(
                    img=self.noise,
                    txt=txt_projected,
                    timestep=sigma_t,
                    guidance=effective_guidance,
                    pooled=pooled_embeds,
                    rope=rope,
                )
                mx.eval(current_output)

                reused, reason = self._teacache_check(teacache_state, current_output, t, steps)

                if reused:
                    noise_pred = teacache_state.last_feature if teacache_state.last_feature is not None else current_output
                    skip_step = True
                else:
                    skip_step = False
                    noise_pred = current_output
            else:
                current_output = self.transformer.predict(
                    img=self.noise,
                    txt=txt_projected,
                    timestep=sigma_t,
                    guidance=effective_guidance,
                    pooled=pooled_embeds,
                    rope=rope,
                )
                mx.eval(current_output)
                noise_pred = current_output
                skip_step = False

            # SeaCache: additional skip check (only if not already skipped)
            if not skip_step and seacache_enabled and t >= warmup_steps and previous_output is not None:
                diff = float(mx.mean(mx.abs(current_output - previous_output)).item())
                if diff < cache_threshold:
                    skip_step = True
                    noise_pred = previous_output
                else:
                    previous_output = current_output

            if skip_step and previous_output is not None:
                noise_pred = previous_output

            if not skip_step:
                previous_output = current_output

            # Euler update: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * noise_pred
            dt = sigma_next - sigma_t
            self.noise = self.noise + noise_pred * dt
            mx.eval(self.noise)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            # Preview
            if self.preview and self.previewer is not None:
                preview_latent = self._denoise_to_latent(self.noise, self.height, self.width, self.output_shape)
                if preview_latent is not None:
                    preview_bytes = self.previewer.decode_latent_to_preview_image(
                        "JPEG", preview_latent
                    )
                    if preview_bytes:
                        import comfy.utils
                        comfy.utils.ProgressBar(steps).update_absolute(t + 1, steps, preview_bytes)

            # Progress logging
            if (t + 1) % 5 == 0 or t == 0:
                cache_tag = ""
                if skip_step:
                    cache_tag = " [cache]"
                if self.kontext and kontext_cache.ready():
                    cache_tag += f" [kontext:{kontext_cache.hits}h/{kontext_cache.stores}s]"
                print(f"[ASDX] Step {t + 1}/{steps} - {step_time:.3f}s{cache_tag}")

        total_time = time.perf_counter() - t_sampling_start

        # Convert result to ComfyUI latent
        out_latent = bridge.mlx_to_comfy_latent(self.noise, self.height, self.width,
                                                {"samples": self.noise})
        out_latent["sdmlx_model_type"] = model_type
        out_latent["sdmlx_model_name"] = "unknown"

        # Memory stats
        mem = bridge.collect_mlx_memory()
        avg_step = sum(step_times) / len(step_times) if step_times else 0

        # Build acceleration summary
        accel_parts = []
        if teacache_state is not None:
            accel_parts.append(f"TeaCache({teacache_state.hits}h/{teacache_state.real_steps}real)")
        if kontext_cache.enabled:
            accel_parts.append(f"Kontext({kontext_cache.hits}h/{kontext_cache.stores}s)")
        if seacache_enabled:
            accel_parts.append("SeaCache")
        accel_str = "+".join(accel_parts) if accel_parts else "standard"

        print(f"[ASDX] Sampling complete: {total_time:.1f}s total, "
              f"{avg_step:.3f}s/step, {mem['peak_gb']:.1f}GB peak, {accel_str}")

        bridge.clear_mlx_cache()

        return out_latent

    @staticmethod
    def _teacache_check(
        state: TeaCacheState,
        current_output: mx.array,
        step: int,
        total_steps: int,
    ) -> tuple[bool, str]:
        """Check if TeaCache allows skipping. Returns (skip, reason)."""
        reused, reason = state.try_reuse(current_output, step, total_steps)
        if reused:
            return True, reason
        return False, "real"

    @staticmethod
    def _prepare_kontext_reference(
        cache: KontextCache,
        reference_latent: dict,
        transformer: Any,
        precision: mx.Dtype,
    ) -> None:
        """Pre-compute reference latent encoding for Kontext conditioning."""
        try:
            import comfy.latent_formats

            latent = reference_latent.get("samples", reference_latent)
            if hasattr(latent, "numpy"):
                ref_np = latent.detach().cpu().float().numpy().astype(np.float32, copy=False)
            else:
                ref_np = np.asarray(latent, dtype=np.float32)

            # Process through FLUX latent format
            model_space = comfy.latent_formats.Flux().process_in(
                torch.from_numpy(ref_np)
            )
            ref_np = model_space.numpy().astype(np.float32, copy=False)

            # Pack reference latent
            batch, channels, ref_h, ref_w = ref_np.shape
            if channels != bridge.FLUX_LATENT_CHANNELS:
                return

            packed = ref_np.reshape(
                batch, channels, ref_h // 2, 2, ref_w // 2, 2
            )
            packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
            packed = packed.reshape(
                batch, (ref_h // 2) * (ref_w // 2), channels * 4
            )

            ref_mlx = mx.array(packed).astype(precision)
            mx.eval(ref_mlx)

            # Compute reference tokens through img_in
            ref_tokens = transformer.img_in(ref_mlx.astype(transformer.dtype))

            # Store reference tokens in cache
            cache.cache["reference"] = (ref_tokens, ref_mlx)

            # Set reference token count (image tokens)
            cache.reference_tokens = ref_tokens.shape[1]

        except Exception as e:
            print(f"[ASDX] Kontext reference prep failed: {e}")

    @staticmethod
    def _update_lora_schedule(model: dict, step: int, total_steps: int) -> None:
        """Update LoRA strength based on schedule."""
        schedule = model.get("lora_schedule")
        if not schedule:
            return

        strength_curve = schedule.get("strength_curve", "linear")
        start = schedule.get("strength_start", 1.0)
        end = schedule.get("strength_end", 0.5)
        middle = schedule.get("strength_middle", 1.0)

        progress = step / max(total_steps, 1)

        if strength_curve == "linear":
            if progress <= 0.5:
                strength = start + (middle - start) * (progress * 2)
            else:
                strength = middle + (end - middle) * ((progress - 0.5) * 2)
        elif strength_curve == "cosine":
            import math
            mid = (start + middle) / 2
            if progress <= 0.5:
                strength = mid + (start - mid) * 0.5 * (1 - math.cos(progress * math.pi))
            else:
                end_val = (middle + end) / 2
                strength = end_val + (end - end_val) * 0.5 * (1 - math.cos((progress - 0.5) * math.pi * 2))
        elif strength_curve == "ease_in_out":
            t = 3 * progress ** 2 - 2 * progress ** 3
            if progress <= 0.5:
                strength = start + (middle - start) * t
            else:
                strength = middle + (end - middle) * t
        else:
            strength = start

        if step % 5 == 0:
            print(f"[ASDX] LoRA schedule: step {step}/{total_steps}, strength={strength:.3f}")

    @staticmethod
    def _create_sigmas(steps: int, model_type: str, width: int = 1024,
                       height: int = 1024) -> list[float]:
        """Create sigma schedule for the given model type.

        FLUX dev uses a shifted log-normal schedule.
        FLUX schnell uses uniform steps.
        """
        if model_type == "schnell":
            return [1.0 - i / steps for i in range(steps + 1)]

        sigmas: list[float] = []
        for i in range(steps + 1):
            t = i / steps
            sigma = 1.0 - t
            area = width * height / (1024 * 1024)
            shift = 0.15 * (area - 1.0)
            sigma = max(0.0, sigma - shift * t)
            sigmas.append(sigma)

        return sigmas

    @staticmethod
    def _denoise_to_latent(
        noise: mx.array,
        height: int,
        width: int,
        output_shape: tuple[int, int],
    ) -> torch.Tensor | None:
        """Convert current MLX latent to a decodable PyTorch latent for preview."""
        try:
            import comfy.latent_formats
            samples = bridge._unpack_flux_latents(noise, height, width)
            samples = comfy.latent_formats.Flux().process_out(samples)
            if torch.backends.mps.is_available():
                return samples.to(device="mps")
            return samples
        except Exception:
            return None

    # ── Mode routing (Phase 2) ────────────────────────────────────────

    def _detect_mode(self) -> SamplerMode:
        """Detect sampling mode from connected inputs.

        Priority: inpainting (image + mask) > img2img (image only) >
        depth (depth_image) > text2img (default).
        """
        if self.mode != "auto":
            return SamplerMode(self.mode)

        has_image = self.image is not None and self.image.numel() > 0
        has_mask = self.mask is not None and self.mask.numel() > 0
        has_depth = (
            self.depth_image is not None and self.depth_image.numel() > 0
        )

        if has_depth:
            return SamplerMode.DEPTH_CONTROL
        if has_image and has_mask:
            return SamplerMode.INPAINTING
        if has_image:
            return SamplerMode.IMAGE_TO_IMAGE
        return SamplerMode.TEXT_TO_IMAGE

    def _encode_image_to_latent(self, image: torch.Tensor) -> mx.array:
        """VAE-encode an image to FLUX latent packed format.

        Returns MLX packed latent [B, NH*NW, 64].
        """
        import comfy.latent_formats

        # Transpose [B, H, W, C] -> [B, C, H, W]
        img = image.permute(0, 3, 1, 2) if image.ndim == 4 else image
        # Normalize [0, 1] -> [-1, 1] for VAE
        img = (img - 0.5) / 0.5

        # Use MLX VAE encoder
        from .mlx_vae import MLXVAE

        vae = MLXVAE()
        latent_mlx = vae.encode(img)

        # Convert to numpy and pack
        latent_np = (
            latent_mlx.detach().cpu().numpy()
            if hasattr(latent_mlx, "detach")
            else np.array(latent_mlx, dtype=np.float32)
        )

        # Pack: [B, C, H, W] -> [B, H/2, W/2, C*4] -> flatten spatial
        batch, channels, latent_h, latent_w = latent_np.shape
        packed = latent_np.reshape(
            batch, channels, latent_h // 2, 2, latent_w // 2, 2
        )
        packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
        packed = packed.reshape(
            batch, (latent_h // 2) * (latent_w // 2), channels * 4
        )

        precision = self.config.mlx_dtype
        packed_mlx = mx.array(packed).astype(precision)
        mx.eval(packed_mlx)
        return packed_mlx

    def _prepare_img2img_noise(self) -> mx.array:
        """Encode input image, add noise at specified strength level.

        Returns MLX packed noise for the denoising loop.
        """
        if self.image is None:
            return self.noise

        # Encode input image to latent
        input_latent = self._encode_image_to_latent(self.image)

        # Add noise at strength level
        import comfy.sample

        # Use ComfyUI's noise addition with sqrt(strength) for proper blending
        noise_strength = math.sqrt(self.image_strength)

        # Convert input latent to packed format matching self.noise
        # Then add noise
        noise_np = np.array(self.noise, dtype=np.float32)
        blended = input_latent * (1.0 - noise_strength) + noise_np * noise_strength

        precision = self.config.mlx_dtype
        return mx.array(blended).astype(precision)

    def _prepare_inpainting_noise(self) -> mx.array:
        """Prepare inpainting noise: encode image, apply mask, add noise.

        The masked region retains the original image (no noise), while
        the unmasked region receives noise. This allows in-painting.
        """
        if self.image is None:
            return self.noise

        # Encode input image to latent
        input_latent = self._encode_image_to_latent(self.image)

        # Prepare mask in packed latent space
        mask_latent = self._prepare_mask_latent()
        if mask_latent is None:
            return self.noise

        # Get raw noise
        noise_np = np.array(self.noise, dtype=np.float32)

        # Apply mask: masked region = original image, unmasked = noise
        # mask=1 means "keep original", mask=0 means "generate new"
        blended = input_latent * mask_latent + noise_np * (1.0 - mask_latent)

        precision = self.config.mlx_dtype
        return mx.array(blended).astype(precision)

    def _prepare_mask_latent(self) -> np.ndarray | None:
        """Convert mask tensor to packed latent-space mask.

        Returns a numpy array in packed format [B, NH*NW, 64] matching
        the noise layout, or None if mask is unavailable.
        """
        if self.mask is None:
            return None

        mask = self.mask
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[0]  # Take first batch item

        h, w = mask.shape[-2:] if mask.ndim == 3 else (mask.shape[0], mask.shape[1])

        # Resize mask to latent space (H/8, W/8)
        latent_h = h // 8
        latent_w = w // 8

        # Resize using bilinear interpolation
        mask_pt = mask.unsqueeze(0).unsqueeze(0) if mask.ndim == 2 else mask
        if mask_pt.shape[-1] != latent_w or mask_pt.shape[-2] != latent_h:
            import torch.nn.functional as F
            mask_pt = F.interpolate(
                mask_pt.float(), size=(latent_h, latent_w), mode="bilinear", align_corners=False
            )
        else:
            mask_pt = mask_pt.float()

        # Blur mask if requested
        if self.mask_blur > 0:
            kernel_size = self.mask_blur if self.mask_blur % 2 == 1 else self.mask_blur + 1
            mask_pt = F.gaussian_blur(mask_pt, kernel_size, sigma=self.mask_blur / 3.0)

        # Squeeze and convert to numpy
        mask_np = mask_pt.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)

        # Expand to packed latent format: [NH, NW] -> [NH, NW, 64]
        latent_np = np.tile(
            mask_np[:, :, np.newaxis, np.newaxis],
            (1, 1, 2, 2)
        ).reshape(latent_h // 2, latent_w // 2, 16 * 4)

        # Pack to [NH/2, NW/2, 64]
        packed = latent_np.reshape(
            latent_h // 2, 2, latent_w // 2, 2, 16 * 4
        )
        packed = np.transpose(packed, (0, 2, 1, 3, 4))
        packed = packed.reshape(
            (latent_h // 2) * (latent_w // 2), 16 * 4
        )

        return packed

    def _prepare_depth_noise(self) -> mx.array:
        """Prepare depth-controlled noise.

        For depth control, the initial noise is the same as text2img,
        but depth conditioning is injected during the transformer forward
        pass (handled by the transformer's depth conditioning path).
        Returns the original noise.
        """
        # Depth conditioning will be handled in the transformer loop
        # via the depth_image passed through the model dict
        return self.noise
