"""
PyTorch <-> MLX Bridge
======================
Conversion utilities optimized for Apple Silicon Unified Memory.

Key patterns:
  - PyTorch tensor -> MLX array:  .detach().cpu().numpy() -> mx.array()
  - MLX array -> PyTorch tensor:  np.array(mx_arr) -> torch.from_numpy()
  - Always call mx.eval() before converting MLX -> NumPy
  - Reuse NumPy buffers when possible to minimize allocations
"""

from __future__ import annotations

import gc
from typing import Any

import mlx.core as mx
import numpy as np
import torch


# ── Constants ─────────────────────────────────────────────────────────

# FLUX uses 16 latent channels; SD1.5/SDXL use 4
FLUX_LATENT_CHANNELS = 16
SDXL_LATENT_CHANNELS = 4

# ComfyUI image format: [B, H, W, C] range [0, 1] float32
# FLUX VAE input/output expects [B, C, H, W] range [-1, 1] float16


# ── PyTorch -> MLX ───────────────────────────────────────────────────

def tensor_to_mlx(tensor: Any, dtype: mx.Dtype | None = None) -> mx.array:
    """Convert a PyTorch tensor to an MLX array.

    Handles both on-CPU and on-device tensors. Calls .detach() and .cpu()
    automatically. Uses float32 intermediate to avoid precision loss during
    the numpy round-trip.
    """
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    out = mx.array(arr)
    if dtype is not None:
        out = out.astype(dtype)
    mx.eval(out)
    return out


def conditioning_to_mlx(
    conditioning: Any,
    precision: mx.Dtype,
) -> tuple[mx.array, mx.array, float | None]:
    """Extract T5 embeddings, pooled CLIP output, and guidance from Comfy conditioning.

    Returns (prompt_embeds, pooled_prompt_embeds, guidance).
    prompt_embeds: [B, T, 4096]  (T5 XXL)
    pooled_prompt_embeds: [B, 768]  (CLIP-L)
    """
    guidance = None

    # Handle wrapper dict (type="flux1")
    if isinstance(conditioning, dict):
        if conditioning.get("type") == "flux1":
            try:
                guidance = float(conditioning["guidance"]) if conditioning.get("guidance") is not None else None
            except Exception:
                guidance = None
            conditioning = conditioning.get("conditioning", conditioning)

        if "cond" in conditioning:
            cond = conditioning.get("cond")
            pooled = conditioning.get("pooled_output", conditioning.get("pooled"))
        else:
            # Standard Comfy conditioning list: [[tensor, {meta}], ...]
            entry = conditioning[0] if conditioning else None
            if entry is None:
                raise RuntimeError("ASDX: positive conditioning is empty.")
            cond = entry[0]
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            pooled = meta.get("pooled_output")
            if "guidance" in meta:
                try:
                    guidance = float(meta["guidance"])
                except Exception:
                    pass
    else:
        raise RuntimeError("ASDX: conditioning must be a list or dict.")

    if pooled is None:
        raise RuntimeError("ASDX: conditioning has no pooled_output.")

    # Convert to numpy then MLX
    cond_np = _to_numpy(cond)
    pooled_np = _to_numpy(pooled)

    # Validate shapes
    if cond_np.ndim != 3 or cond_np.shape[-1] != 4096:
        raise RuntimeError(
            f"ASDX: expected T5 embeddings [B,T,4096], got {cond_np.shape}"
        )
    if pooled_np.ndim != 2 or pooled_np.shape[-1] != 768:
        raise RuntimeError(
            f"ASDX: expected pooled CLIP [B,768], got {pooled_np.shape}"
        )
    if cond_np.shape[0] != 1 or pooled_np.shape[0] != 1:
        raise RuntimeError("ASDX: currently supports batch_size=1 only.")

    prompt_embeds = mx.array(cond_np).astype(precision)
    pooled_prompt_embeds = mx.array(pooled_np).astype(precision)
    mx.eval(prompt_embeds, pooled_prompt_embeds)

    return prompt_embeds, pooled_prompt_embeds, guidance


def _to_numpy(value: Any) -> np.ndarray:
    """Convert any tensor-like object to a NumPy float32 array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


# ── MLX -> PyTorch ───────────────────────────────────────────────────

def mlx_to_comfy_image(decoded: mx.array) -> torch.Tensor:
    """Convert MLX decoded VAE output to ComfyUI IMAGE format [B,H,W,C].

    VAE output is [B, C, H, W] in range [-1, 1]. This normalizes to [0, 1]
    and transposes to ComfyUI convention.
    """
    image = mx.clip((decoded.astype(mx.float32) / 2.0) + 0.5, 0.0, 1.0)
    image = mx.transpose(image, (0, 2, 3, 1))
    mx.eval(image)
    return torch.from_numpy(np.array(image, dtype=np.float32))


def mlx_to_comfy_latent(
    latents: mx.array,
    height: int,
    width: int,
    template: dict[str, Any],
) -> dict[str, Any]:
    """Convert MLX latent to ComfyUI LATENT dict.

    FLUX latents are packed as [B, H_lat*W_lat, 64] where 64 = 16*2*2
    (16 channels * 2*2 patch). Unpack back to [B, 16, H/8, W/8].
    """
    samples = _unpack_flux_latents(latents, height, width)
    out = dict(template)
    out["samples"] = samples
    return out


def _unpack_flux_latents(
    latents: mx.array,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unpack packed FLUX latent [B, NW*NH, 64] -> [B, 16, H/8, W/8]."""
    latent_h = height // 8
    latent_w = width // 8

    # Reshape: [B, NH, NW, 16, 2, 2] -> [B, 16, NH, 2, NW, 2] -> [B, 16, NH*2, NW*2]
    unpacked = mx.reshape(latents, (1, latent_h // 2, latent_w // 2, 16, 2, 2))
    unpacked = mx.transpose(unpacked, (0, 3, 1, 4, 2, 5))
    unpacked = mx.reshape(unpacked, (1, 16, latent_h // 2 * 2, latent_w // 2 * 2))
    unpacked = unpacked.astype(mx.float32)
    mx.eval(unpacked)

    samples = torch.from_numpy(np.array(unpacked, dtype=np.float32))
    return samples


# ── Noise preparation ────────────────────────────────────────────────

def prepare_noise_from_latent(
    latent: dict[str, Any],
    seed: int,
    precision: mx.Dtype,
) -> tuple[mx.array, int, int, tuple[int, int]]:
    """Prepare initial noise from a Comfy latent, converted to MLX.

    FLUX packs latents as [B, H/8, W/8, 16] -> packed to [B, NH*NW, 64].
    Returns (noise, height, width, output_shape).
    """
    if "samples" not in latent:
        raise RuntimeError("ASDX: latent must be a Comfy LATENT with 'samples'.")

    samples = latent["samples"]
    if tuple(samples.shape)[1] != FLUX_LATENT_CHANNELS:
        raise RuntimeError(
            f"ASDX: needs 16-channel FLUX latent, got {tuple(samples.shape)}"
        )

    # ComfyUI noise
    import comfy.sample
    batch_inds = latent.get("batch_index") if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(samples, int(seed), batch_inds)

    # Convert to numpy and pack
    noise_np = noise.detach().cpu().float().numpy().astype(np.float32, copy=False)

    # Pad to even dimensions if needed
    h, w = noise_np.shape[-2], noise_np.shape[-1]
    if h % 2 != 0 or w % 2 != 0:
        pad_h = h % 2
        pad_w = w % 2
        noise_np = np.pad(noise_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    height = noise_np.shape[-2] * 8
    width = noise_np.shape[-1] * 8

    # Pack: [B, C, H, W] -> [B, H/2, W/2, C*4] -> flatten spatial
    batch, channels, latent_h, latent_w = noise_np.shape
    packed = noise_np.reshape(batch, channels, latent_h // 2, 2, latent_w // 2, 2)
    packed = np.transpose(packed, (0, 2, 4, 1, 3, 5))
    packed = packed.reshape(batch, (latent_h // 2) * (latent_w // 2), channels * 4)

    noise_mlx = mx.array(packed).astype(precision)
    mx.eval(noise_mlx)

    output_shape = (int(samples.shape[-2]), int(samples.shape[-1]))
    return noise_mlx, height, width, output_shape


# ── Memory management ────────────────────────────────────────────────

def collect_mlx_memory() -> dict[str, float]:
    """Return current MLX memory stats in GB."""
    active = mx.get_active_memory() / (1024 ** 3)
    cached = mx.get_cache_memory() / (1024 ** 3)
    peak = mx.get_peak_memory() / (1024 ** 3)
    return {"active_gb": round(active, 2), "cache_gb": round(cached, 2), "peak_gb": round(peak, 2)}


def clear_mlx_cache() -> None:
    """Clear MLX constant cache and trigger GC. Call between major phases."""
    mx.clear_cache()
    gc.collect()
    # Also clear MPS cache if available
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def set_mlx_cache_limit_gb(gb: float) -> None:
    """Set MLX constant cache limit in GB."""
    mx.set_cache_limit(int(gb * 1024 ** 3))
