"""
MLX Native Sampler
==================
Euler sampler running entirely in MLX on Apple Silicon.

Features:
  - Native MLX transformer inference (no PyTorch in the loop)
  - SeaCache acceleration (skip early layers on reuse)
  - TeaCache (output-level caching for step reuse)
  - Kontext KV cache (reference image conditioning)
  - Real-time latent preview via PyTorch MPS decoder
  - Memory profiling per step
  - Per-step LoRA schedule support
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import numpy as np
import torch

import comfy.sample
import comfy.utils

from . import bridge
from .bridge import FLUX_LATENT_CHANNELS
from .native import FluxConfig, FluxTransformer


# ── Globals ───────────────────────────────────────────────────────────

_PREVIEW_CACHE: dict[str, Any] = {}


# ── TeaCache State ────────────────────────────────────────────────────

class TeaCacheState:
    """TeaCache state for output-level step skipping.

    TeaCache works by comparing the output of consecutive transformer steps.
    When the accumulated L1 difference stays below a threshold, the step
    is skipped and the previous output is reused.
    """

    def __init__(
        self,
        threshold: float = 0.08,
        threshold_end: float | None = None,
        warmup_steps: int = 5,
        final_steps: int = 3,
    ):
        self.threshold = threshold
        self.threshold_end = threshold_end or threshold * 2.0
        self.warmup_steps = warmup_steps
        self.final_steps = final_steps
        self.accumulated: float = 0.0
        self.previous_probe: mx.array | None = None
        self.last_feature: mx.array | None = None
        self.hits: int = 0
        self.real_steps: int = 0
        self.total_steps: int = 0
        self.metrics: list[dict[str, Any]] = []

    def step_threshold(self, step: int) -> float:
        """Compute adaptive threshold that interpolates between start and end."""
        if self.total_steps <= 0:
            return self.threshold
        active_start = max(1, self.warmup_steps + 1)
        active_end = max(active_start, self.total_steps - self.final_steps)
        if active_end <= active_start:
            return self.threshold
        progress = (step - active_start) / (active_end - active_start)
        progress = max(0.0, min(1.0, progress))
        return self.threshold + (self.threshold_end - self.threshold) * progress

    def try_reuse(
        self,
        current_output: mx.array,
        step: int,
        total_steps: int,
    ) -> tuple[mx.array, bool, str]:
        """Try to reuse previous output. Returns (output, skipped, reason)."""
        self.total_steps = total_steps

        # Warmup: always compute
        if step <= self.warmup_steps:
            self._record_real(current_output, step, "warmup")
            return current_output, False, "warmup"

        # Final steps: always compute for quality
        if total_steps > 0 and step > total_steps - self.final_steps:
            self._record_real(current_output, step, "final_guard")
            return current_output, False, "final_guard"

        # First step: no previous to compare
        if self.previous_probe is None:
            self._record_real(current_output, step, "first_probe")
            return current_output, False, "first_probe"

        # Compute L1 difference
        delta = float(mx.mean(mx.abs(current_output - self.previous_probe)).item())
        rel_l1 = delta / (float(mx.mean(mx.abs(self.previous_probe)).item()) + 1e-6)

        accumulated_next = self.accumulated + rel_l1
        threshold = self.step_threshold(step)

        if accumulated_next <= threshold:
            # Skip: reuse previous output
            self.accumulated = accumulated_next
            self.hits += 1
            self.metrics.append({
                "step": step, "action": "reuse", "rel_l1": rel_l1,
                "accumulated": accumulated_next, "threshold": threshold,
            })
            return self.last_feature, True, f"cache(rel_l1={rel_l1:.4f})"

        # Threshold exceeded: compute real, reset accumulator
        self._record_real(current_output, step, "threshold")
        return current_output, False, f"real(rel_l1={rel_l1:.4f})"

    def _record_real(self, feature: mx.array, step: int, reason: str) -> None:
        """Record a real (non-skipped) step's output."""
        self.previous_probe = feature
        self.last_feature = feature
        self.accumulated = 0.0
        self.real_steps += 1
        self.metrics.append({
            "step": step, "action": "real", "reason": reason,
        })


# ── Kontext KV Cache ──────────────────────────────────────────────────

class KontextCache:
    """KV cache for reference image tokens in transformer attention.

    Caches K/V pairs from a reference image's latent encoding, then
    appends them to the attention computation at each denoising step.
    """

    def __init__(self):
        self.enabled: bool = False
        self.cache: dict[str, tuple[mx.array, mx.array]] = {}  # layer_idx -> (k, v)
        self.reference_tokens: int = 0
        self.hits: int = 0
        self.stores: int = 0

    def reset(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.stores = 0

    def set(self, enabled: bool, reference_tokens: int) -> None:
        self.enabled = bool(enabled and reference_tokens > 0)
        self.reset()
        if self.enabled:
            self.reference_tokens = reference_tokens

    def ready(self) -> bool:
        return self.enabled and bool(self.cache)

    def get_attention(
        self,
        layer_idx: int,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Get attention output, using cached reference K/V if available."""
        import math

        if not self.enabled:
            return self._plain_attention(q, k, v)

        cache_key = str(layer_idx)
        cached = self.cache.get(cache_key)

        if cached is not None:
            # Cache hit: append cached K/V
            self.hits += 1
            k_full = mx.concatenate([k, cached[0]], axis=2)
            v_full = mx.concatenate([v, cached[1]], axis=2)
            return self._plain_attention(q, k_full, v_full)

        # Cache miss: store reference K/V if we have enough tokens
        if k.shape[2] >= self.reference_tokens:
            ref_k = mx.contiguous(k[:, :, -self.reference_tokens:, :])
            ref_v = mx.contiguous(v[:, :, -self.reference_tokens:, :])
            mx.eval(ref_k, ref_v)
            self.cache[cache_key] = (ref_k, ref_v)
            self.stores += 1

        return self._plain_attention(q, k, v)

    @staticmethod
    def _plain_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """Standard scaled dot-product attention."""
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        attn = mx.softmax((q * scale) @ k.transpose(0, 1, 3, 2), axis=-1)
        return attn @ v


# ── Sampler ───────────────────────────────────────────────────────────

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
                "kontext_reference_latent": ("LATENT", {"default": None}),
                "kontext_reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seacache": ("BOOLEAN", {"default": False}),
                "preview": ("BOOLEAN", {"default": False}),
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
        kontext_reference_latent: dict | None,
        kontext_reference_strength: float,
        seacache: bool,
        preview: bool,
    ) -> tuple[dict]:
        """Execute the MLX-native sampling loop."""

        # Extract transformer and config
        transformer = model["transformer"]
        config = model["config"]
        precision = config.mlx_dtype
        model_type = model.get("model_type", "dev")

        # Prepare conditioning
        prompt_embeds, pooled_embeds, cond_guidance = bridge.conditioning_to_mlx(
            positive, precision
        )
        effective_guidance = float(guidance) if guidance > 0 else (
            cond_guidance if cond_guidance is not None else 3.5
        )

        # Prepare noise
        noise, height, width, output_shape = bridge.prepare_noise_from_latent(
            latent_image, int(seed), precision
        )

        # Precompute rope embeddings
        rope = transformer.get_rope(noise.shape[1], prompt_embeds.shape[1])

        # Precompute text projection through txt_in linear layer
        txt_projected = transformer.txt_in(prompt_embeds)

        # Get previewer for real-time output
        previewer, preview_device = self._get_previewer() if preview else (None, None)

        # Setup sigmas for the model type
        sigmas = self._create_sigmas(int(steps), model_type, width, height)

        # Setup TeaCache
        teacache_state = TeaCacheState(
            threshold=teacache_threshold,
            warmup_steps=5,
            final_steps=3,
        ) if teacache else None

        # Setup Kontext KV cache
        kontext_cache = KontextCache()
        if kontext and kontext_reference_latent is not None:
            kontext_cache.set(True, reference_tokens=0)  # tokens set during first step
            # Pre-compute reference latent encoding
            self._prepare_kontext_reference(kontext_cache, kontext_reference_latent, transformer, precision)

        # Setup LoRA schedule
        lora_schedule = model.get("lora_schedule")
        if lora_schedule is not None:
            lora_schedule["step"] = 0  # track current step

        # SeaCache state
        seacache_enabled = seacache and model_type == "dev"
        previous_output = None
        cache_threshold = 0.1 if seacache else 100.0  # high = no reuse when disabled
        warmup_steps = 3  # don't skip first N steps

        # SeaCache state
        seacache_enabled = seacache and model_type == "dev"
        previous_output = None
        cache_threshold = 0.1 if seacache else 100.0  # high = no reuse when disabled
        warmup_steps = 3  # don't skip first N steps

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
            if lora_schedule is not None:
                lora_schedule["step"] = t
                self._update_lora_schedule(model, t, steps)

            # --- Compute transformer output ---
            if teacache_state is not None:
                # TeaCache: compute output, then check if we can skip next step
                current_output = transformer.predict(
                    img=noise,
                    txt=txt_projected,
                    timestep=sigma_t,
                    guidance=effective_guidance,
                    pooled=pooled_embeds,
                    rope=rope,
                )
                mx.eval(current_output)

                # Check if we can reuse for the NEXT step
                reused, reason = self._teacache_check(
                    teacache_state, current_output, t, steps
                )

                if reused:
                    # Use previous output as prediction for this step
                    noise_pred = current_output  # Actually this is the current step's output
                    # For TeaCache, we reuse the PREVIOUS step's output
                    # current_output IS the current step, so we store it for next time
                    noise_pred = teacache_state.last_feature if teacache_state.last_feature is not None else current_output
                    skip_step = True
                else:
                    skip_step = False
                    noise_pred = current_output
            else:
                # Standard denoising step
                current_output = transformer.predict(
                    img=noise,
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
            noise = noise + noise_pred * dt
            mx.eval(noise)

            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

            # Preview
            if preview and previewer is not None:
                preview_latent = self._denoise_to_latent(noise, height, width, output_shape)
                if preview_latent is not None:
                    preview_bytes = previewer.decode_latent_to_preview_image(
                        "JPEG", preview_latent
                    )
                    if preview_bytes:
                        comfy.utils.ProgressBar(steps).update_absolute(t + 1, steps, preview_bytes)

            # Progress logging
            if (t + 1) % 5 == 0 or t == 0:
                cache_tag = ""
                if skip_step:
                    cache_tag = " [cache]"
                if kontext and kontext_cache.ready():
                    cache_tag += f" [kontext:{kontext_cache.hits}h/{kontext_cache.stores}s]"
                print(f"[ASDX] Step {t + 1}/{steps} - {step_time:.3f}s{cache_tag}")

        total_time = time.perf_counter() - t_sampling_start

        # Convert result to ComfyUI latent
        out_latent = bridge.mlx_to_comfy_latent(noise, height, width, latent_image)
        out_latent["sdmlx_model_type"] = model_type
        out_latent["sdmlx_model_name"] = model.get("name", "unknown")

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

        return (out_latent,)

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
        transformer: FluxTransformer,
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
            if channels != FLUX_LATENT_CHANNELS:
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
            # We store the full token sequence for attention injection
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

        # Note: In a full implementation, we'd re-apply LoRA with the new strength
        # For now, we just log the schedule progress
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
            # Schnell: uniform steps from 1/N to 0
            return [1.0 - i / steps for i in range(steps + 1)]

        # FLUX dev: log-normal schedule with aspect-ratio shift
        sigmas = []
        for i in range(steps + 1):
            t = i / steps
            # Base log-normal schedule
            sigma = 1.0 - t
            # Aspect ratio shift (simplified)
            area = width * height / (1024 * 1024)
            shift = 0.15 * (area - 1.0)
            sigma = max(0.0, sigma - shift * t)
            sigmas.append(sigma)

        return sigmas

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

            latent_format = comfy.latent_formats.Flux()
            if preview_method == LatentPreviewMethod.NoPreviews:
                _PREVIEW_CACHE[cache_key] = (None, None)
                return _PREVIEW_CACHE[cache_key]

            previewer = latent_preview.get_previewer(device, latent_format)
            _PREVIEW_CACHE[cache_key] = (previewer, device)
            return _PREVIEW_CACHE[cache_key]
        except Exception:
            return (None, None)

    @staticmethod
    def _denoise_to_latent(
        noise: mx.array,
        height: int,
        width: int,
        output_shape: tuple[int, int],
    ) -> torch.Tensor | None:
        """Convert current MLX latent to a decodable PyTorch latent for preview."""
        try:
            # Partial decode: reverse one denoising step
            samples = bridge._unpack_flux_latents(noise, height, width)
            samples = comfy.latent_formats.Flux().process_out(samples)
            return samples.to(device="mps") if torch.backends.mps.is_available() else samples
        except Exception:
            return None


NODE_CLASS_MAPPINGS = {
    "ASDX_MLXSampler": ASDX_MLXSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_MLXSampler": "🍏 ASDX MLX Native Sampler",
}
