"""
Diffusion Model Loader
======================
Loads FLUX.1 checkpoints into MLX-native transformers.

Features:
  - Automatic checkpoint type detection (dev vs schnell)
  - Quantization support: dense, FP8, GGUF (via sdmlx native)
  - Model weight caching to avoid reload
  - Memory profiling on load
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch

from . import bridge
from .capability import CAPABILITY_PROFILES, CapabilityProfile, _resolve_capability_from_path
from .native import FluxConfig, FluxTransformer, load_transformer


# ── Globals ───────────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, dict[str, Any]] = {}

# Composite cache key components (set by LoRA/ControlNet nodes)
_MODEL_EXTRA_KEYS: dict[str, str] = {}


def _build_cache_key(base_key: str, extra: dict[str, str] | None = None) -> str:
    """Build a composite cache key matching mflux-AnyModel pattern.

    Combines the base model key with optional LoRA, ControlNet, and
    base model identifiers for fine-grained cache management.
    """
    if extra:
        parts = [base_key]
        for k in ("lora", "controlnet", "base_model"):
            if k in extra and extra[k]:
                parts.append(f"{k}:{extra[k]}")
        return ":".join(parts)
    return base_key
_TYPE_HINTS = {
    "schnell": "schnell",
    "dev": "dev",
    "kontext": "dev",
}

_KREA2_HINTS = {
    "krea2": "krea2",
    "krea": "krea2",
}

_SDXL_HINTS = {
    "sdxl": "sdxl",
    "illustrious": "sdxl",
    "pony": "sdxl",
    "noobai": "sdxl",
}

_ZIMAGE_HINTS = {
    "zimage": "zimage",
    "z_image": "zimage",
    "z-image": "zimage",
}

_FLUX2_HINTS = {
    "klein": "flux2",
    "flux.2": "flux2",
    "flux2": "flux2",
    "flux_2": "flux2",
}


def _detect_model_type(path: Path) -> str:
    """Detect model type from filename, falling back to checkpoint key
    inspection when the filename gives no hint.

    Filename hints are checked first (cheap, matches established
    convention). `_FLUX2_HINTS` is checked BEFORE the generic
    `_TYPE_HINTS` (schnell/dev/kontext): Flux2-D's real filename on this
    machine (`flux2_dev_fp8mixed.safetensors`) contains "dev" too, which
    would otherwise misroute it to the FLUX.1-dev architecture (the exact
    class of bug `_TYPE_HINTS["klein"]="schnell"` used to be, for Klein).
    Many SDXL finetunes (Illustrious/Pony/NoobAI merges in particular)
    don't include an obvious marker in their filename, so if nothing
    matches, peek at the checkpoint's own tensor keys (header-only read via
    safetensors, no weight data loaded) — SDXL's conv UNet has a
    structurally distinctive `input_blocks.` key that FLUX/Krea2 never have,
    Z-Image has an equally distinctive `noise_refiner.` key, and Flux2 has
    `double_stream_modulation_img.` (comfy's own detection marker for this
    exact branch — see `comfy/model_detection.py:237`).
    """
    name = path.name.lower()
    for hint in _KREA2_HINTS:
        if hint in name:
            return "krea2"
    for hint in _SDXL_HINTS:
        if hint in name:
            return "sdxl"
    for hint in _ZIMAGE_HINTS:
        if hint in name:
            return "zimage_turbo" if "turbo" in name else "zimage"
    for hint in _FLUX2_HINTS:
        if hint in name:
            return "flux2"
    for hint in _TYPE_HINTS:
        if hint in name:
            return _TYPE_HINTS[hint]
    return _detect_model_type_from_keys(path)


def _detect_model_type_from_keys(path: Path) -> str:
    """Fallback: distinguish SDXL/Z-Image/Flux2 from FLUX.1/Krea2 by checkpoint tensor keys."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
    except Exception as e:
        print(f"[ASDX] Model type key-detection failed ({e}), defaulting to 'dev'")
        return "dev"

    if any("diffusion_model.input_blocks." in k or k.startswith("input_blocks.") for k in keys):
        return "sdxl"
    if any("noise_refiner." in k for k in keys):
        return "zimage"
    if any("double_stream_modulation_img." in k for k in keys):
        return "flux2"
    if any("txtfusion." in k for k in keys):
        return "krea2"
    return "dev"


def _load_transformer_for_type(
    path: Path, model_type: str, dtype: str
):
    """Load transformer weights based on detected model type.

    Returns (transformer, config) tuple.
    """
    if model_type == "krea2":
        from .native.krea2 import (
            Krea2Config,
            SingleStreamDiT,
            load_krea2_transformer,
        )
        config = Krea2Config(dtype=dtype)
        transformer = load_krea2_transformer(path, dtype=dtype)
        return transformer, config
    elif model_type == "sdxl":
        from .native.sdxl import SDXLConfig, load_sdxl_unet
        config = SDXLConfig(dtype=dtype)
        transformer = load_sdxl_unet(path, dtype=dtype)
        return transformer, config
    elif model_type in ("zimage", "zimage_turbo"):
        from .native.zimage import ZImageConfig, load_zimage_transformer
        config = ZImageConfig(dtype=dtype)
        transformer = load_zimage_transformer(path, dtype=dtype)
        return transformer, config
    elif model_type == "flux2":
        from .native.flux2 import load_flux2_transformer
        transformer = load_flux2_transformer(path, dtype=dtype)
        # Unlike the other families, Flux2's config is DETECTED from the
        # checkpoint (hidden_size/depth/guidance_embed differ between Klein
        # and Flux2-D) — reuse the config load_flux2_transformer already
        # derived, don't construct a fresh default one.
        return transformer, transformer.config
    else:
        # FLUX.1 path
        guidance_embed = model_type == "dev"
        config = FluxConfig(dtype=dtype, guidance_embed=guidance_embed)
        transformer = load_transformer(path, dtype=dtype)
        return transformer, config


_MODEL_TYPE_CAPABILITY = {
    "sdxl": "sdxl_base",
    "zimage": "zimage_base",
    "zimage_turbo": "zimage_turbo",
    "flux2": "flux2_klein",
}


def _capability_for_model_type(model_type: str, path: Path) -> CapabilityProfile:
    """Resolve a capability profile, preferring the already-known `model_type`
    (from `_detect_model_type`, which includes a content-based fallback for
    ambiguous filenames) over re-guessing from the filename alone — avoids
    the two detection systems disagreeing for checkpoints whose name gave
    no hint (e.g. Illustrious/Pony/NoobAI SDXL finetunes)."""
    profile_key = _MODEL_TYPE_CAPABILITY.get(model_type)
    if profile_key is not None:
        return CAPABILITY_PROFILES[profile_key]
    return _resolve_capability_from_path(path)


def _model_type_from_path(path: Path) -> str:
    """Infer model type from filename."""
    name = path.name.lower()
    for hint, model_type in _TYPE_HINTS.items():
        if hint in name:
            return model_type
    return "dev"  # default


# ── Node ──────────────────────────────────────────────────────────────

class ASDX_DiffusionLoader:
    """Load a FLUX.1 diffusion model checkpoint into MLX.

    Reads the checkpoint, creates a FluxTransformer, and caches it.
    The returned model object is passed to the sampler node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls._get_models(),),
                "precision": (["float16", "bfloat16"], {"default": "float16"}),
            },
            "optional": {
                "lora": ("ASDX_LORA", {"default": None}),
                "controlnet": ("ASDX_CONTROLNET", {"default": None}),
                "base_model": ("ASDX_MODEL", {"default": None}),
                "low_memory_mode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("asdx_model",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "ASDX/Loaders"

    @staticmethod
    def _get_models() -> list[str]:
        """Get list of available diffusion models."""
        try:
            import folder_paths
            models = []
            for folder in ("diffusion_models", "unet"):
                try:
                    models.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if models:
                return models
        except Exception:
            pass
        return ["flux1-dev-fp16.safetensors"]

    def load(
        self,
        model_name: str,
        precision: str,
        unique_id: int | None = None,
        lora: str | None = None,
        controlnet: str | None = None,
        base_model: str | None = None,
        low_memory_mode: bool = False,
    ) -> tuple[dict]:
        t0 = time.perf_counter()

        # Build composite cache key
        extra: dict[str, str] = {}
        if lora:
            extra["lora"] = lora
        if controlnet:
            extra["controlnet"] = controlnet
        if base_model:
            extra["base_model"] = base_model
        base_key = f"{model_name}:{precision}"
        cache_key = _build_cache_key(base_key, extra if extra else None)

        if cache_key in _MODEL_CACHE:
            cached = _MODEL_CACHE[cache_key]
            print(f"[ASDX] Model cache hit: {model_name} ({precision})")
            return (cached,)

        # Find model file
        path = self._resolve_model_path(model_name)
        model_type = _detect_model_type(path)

        # Resolve capability profile (see _capability_for_model_type).
        capability = _capability_for_model_type(model_type, path)

        # Load transformer based on model type (FLUX.1, Krea2, SDXL, or Z-Image)
        transformer, config = _load_transformer_for_type(
            path, model_type, precision
        )

        # _load_safetensors() upcasts BF16 checkpoint tensors to float32 before
        # the loader casts them down to the requested precision; the discarded
        # float32 buffers land in MLX's cache (freed but not returned to the
        # OS) rather than active memory. Release them now instead of letting
        # them sit alongside the real active weights for the rest of the run.
        bridge.clear_mlx_cache()

        # Create model descriptor with capability profile
        model_desc = {
            "type": "asdx_model",
            "name": model_name,
            "path": str(path),
            "transformer": transformer,
            "config": config,
            "model_type": model_type,
            "precision": precision,
            "capability": capability,
            "load_time": 0.0,
            "low_memory_mode": low_memory_mode,
        }

        load_time = time.perf_counter() - t0
        model_desc["load_time"] = load_time

        mem = bridge.collect_mlx_memory()
        print(f"[ASDX] Loaded {model_name} in {load_time:.1f}s "
              f"(type={model_type}, precision={precision}, "
              f"mem={mem['active_gb']:.1f}GB active, {mem['cache_gb']:.1f}GB cache)")

        _MODEL_CACHE[cache_key] = model_desc
        return (model_desc,)

    @staticmethod
    def _resolve_model_path(name: str) -> Path:
        """Resolve model name to a file path."""
        try:
            import folder_paths
            for folder in ("diffusion_models", "unet"):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: check common locations
        for candidate in (
            Path.home() / "models" / "diffusion_models" / name,
            Path.home() / "ComfyUI" / "models" / "diffusion_models" / name,
        ):
            if candidate.exists():
                return candidate
        return Path(name)


# ── Checkpoint Loader ────────────────────────────────────────────────────

class ASDX_CheckpointLoader:
    """Load a full checkpoint (VAE + CLIP + Diffusion) into MLX.

    Reads the checkpoint, creates MLX model handles for the diffusion
    transformer, text encoders, and VAE. Returns handles that can be
    passed to the sampler and conditioning nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (cls._get_checkpoints(),),
                "precision": (["float16", "bfloat16"], {"default": "float16"}),
            },
        }

    RETURN_TYPES = ("asdx_model", "mlx_clip", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "ASDX/Loaders"

    @staticmethod
    def _get_checkpoints() -> list[str]:
        """Get list of available checkpoint files."""
        try:
            import folder_paths
            checkpoints = []
            for folder in ("checkpoints", "diffusion_models"):
                try:
                    checkpoints.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if checkpoints:
                return checkpoints
        except Exception:
            pass
        return ["flux1-dev-fp16.safetensors"]

    def load(self, ckpt_name: str, precision: str) -> tuple[dict, dict, Any]:
        t0 = time.perf_counter()

        # Resolve checkpoint path
        path = self._resolve_checkpoint_path(ckpt_name)
        model_type = _detect_model_type(path)

        # Load diffusion model based on type
        transformer, config = _load_transformer_for_type(
            path, model_type, precision
        )

        # See ASDX_DiffusionLoader.load() — release the float32 buffers
        # _load_safetensors()/the dtype cast leave behind in MLX's cache.
        bridge.clear_mlx_cache()

        # Resolve capability profile (see _capability_for_model_type — this
        # loader is the one most likely to see merged SDXL/Illustrious/Pony
        # checkpoints, whose filenames are often ambiguous).
        capability = _capability_for_model_type(model_type, path)

        # Create model descriptor
        model_desc = {
            "type": "asdx_model",
            "name": ckpt_name,
            "path": str(path),
            "transformer": transformer,
            "config": config,
            "model_type": model_type,
            "precision": precision,
            "capability": capability,
        }

        # Real comfy.sd.CLIP + comfy.sd.VAE, extracted from the same checkpoint
        # file in one pass — NOT placeholders. The diffusion transformer is
        # loaded separately above via our own MLX-native reader, so
        # `output_model=False` skips the (expensive, redundant) PyTorch UNet
        # build; only the CLIP/VAE-prefixed tensors are used. `clip` must be a
        # real `comfy.sd.CLIP` (ASDX_CLIPTextEncode does `isinstance(mlx_clip,
        # comfy.sd.CLIP)`), and `vae` must be a real "VAE"-typed comfy object
        # (ASDX_VAEDecode's `vae` input socket only accepts the standard
        # ComfyUI "VAE" type, and its decode path calls `vae.decode()`).
        # Bonus over the standalone ASDX_DualCLIPLoader path: the text-encoder
        # architecture (e.g. Klein's Qwen3-4B vs Qwen3-8B vs Flux2-D's
        # Mistral3-24B) is detected from the checkpoint's own embedded CLIP
        # weights (`model_config.clip_target(state_dict)`), not from a
        # user-selected `clip_type` dropdown — sidesteps the Klein-4B
        # misrouting footgun that dropdown has when clip_type is left at its
        # default (see Phase F notes in the multi-model plan).
        import comfy.sd
        _, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            str(path), output_model=False, output_clip=True,
            output_vae=True, output_clipvision=False,
        )
        clip_desc = clip

        load_time = time.perf_counter() - t0
        mem = bridge.collect_mlx_memory()
        print(f"[ASDX] Checkpoint loaded: {ckpt_name} in {load_time:.1f}s "
              f"(type={model_type}, precision={precision}, "
              f"mem={mem['active_gb']:.1f}GB active, {mem['cache_gb']:.1f}GB cache)")

        return (model_desc, clip_desc, vae)

    @staticmethod
    def _resolve_checkpoint_path(name: str) -> Path:
        """Resolve checkpoint name to a file path."""
        try:
            import folder_paths
            for folder in ("checkpoints", "diffusion_models"):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: check common locations
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "checkpoints" / name,
            Path.home() / "ComfyUI" / "models" / "diffusion_models" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ASDX_DiffusionLoader": ASDX_DiffusionLoader,
    "ASDX_CheckpointLoader": ASDX_CheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_DiffusionLoader": "🍏 ASDX Diffusion Loader",
    "ASDX_CheckpointLoader": "🍏 ASDX Checkpoint Loader",
}
