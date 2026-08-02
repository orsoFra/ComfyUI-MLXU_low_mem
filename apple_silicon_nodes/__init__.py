"""
Apple Silicon Diffusion Nodes
=============================
Custom ComfyUI nodes optimized for Apple Silicon (M1/M2/M3/M4/M5)
using MLX native inference with zero-copy Unified Memory semantics.

Key design principles:
  - MLX native for diffusion backbones (transformer/unet)
  - PyTorch MPS for ComfyUI compatibility where conversion is impractical
  - Strategic mx.eval() / mx.clear_cache() at memory hotspots
  - float16/bfloat16 native on Apple Silicon (avoid torch.float16 MPS NaN issues)

Advanced features:
  - LoRA runtime loading (standard A/B and ComfyUI diff format)
  - TeaCache acceleration (output-level step skipping)
  - SeaCache acceleration (residual-based step skipping)
  - Kontext KV cache (reference image conditioning)
  - ControlNet Union (8 control types)
  - IP-Adapter (cross-attention injection)
"""

from __future__ import annotations

__version__ = "0.2.0"

_WEB_DIRECTORY = "web"

# ── Node display names ────────────────────────────────────────────────
_DISPLAY = {
    # Core
    "ASDX_DiffusionLoader": "🍏 ASDX Diffusion Loader",
    "ASDX_CheckpointLoader": "🍏 ASDX Checkpoint Loader",
    "ASDX_DualCLIPLoader": "🍏 ASDX Dual CLIP Loader",
    "ASDX_CLIPLoader": "🍏 ASDX CLIP Loader",
    "ASDX_CLIPTextEncode": "🍏 ASDX CLIP Text Encode",
    "ASDX_MLXSampler": "🍏 ASDX MLX Native Sampler",
    "ASDX_VAEDecode": "🍏 ASDX VAE Decode (MLX)",
    "ASDX_VAEEncode": "🍏 ASDX VAE Encode (MLX)",
    "ASDX_EmptyFLUXLatent": "🍏 ASDX Empty FLUX Latent",
    "ASDX_MemoryProfiler": "🍏 ASDX Memory Profiler",
    # Conditioning
    "ASDX_ConditioningMerger": "🍏 ASDX Conditioning Merger",
    # LoRA
    "ASDX_LoraLoader": "🍏 ASDX LoRA Loader",
    "ASDX_MultiLoraLoader": "🍏 ASDX Multi LoRA Loader",
    "ASDX_LoraSchedule": "🍏 ASDX LoRA Schedule",
    # ControlNet
    "ASDX_ControlNetUnionLoader": "🍏 ASDX ControlNet Union Loader",
    "ASDX_ApplyControlNet": "🍏 ASDX Apply ControlNet",
    # IP-Adapter
    "ASDX_IPAdapterLoader": "🍏 ASDX IP-Adapter Loader",
    "ASDX_IPAdapterCLIPVisionEncode": "🍏 ASDX CLIP Vision Encode",
    "ASDX_ApplyIPAdapter": "🍏 ASDX Apply IP-Adapter",
    # Image Chain
    "ASDX_ImageToLatent": "🍏 ASDX Image → Latent",
    "ASDX_MaskFromImage": "🍏 ASDX Mask From Image",
    "ASDX_MaskBlur": "🍏 ASDX Mask Blur",
    "ASDX_ImageCompositor": "🍏 ASDX Image Compositor",
    # Depth
    "ASDX_DepthMap": "🍏 ASDX Depth Map",
    # Utilities
    "ASDX_LivePreview": "🍏 ASDX Live Preview",
}

# ── Lazy import with graceful fallback ────────────────────────────────
try:
    from .loader import NODE_CLASS_MAPPINGS as _loader_maps
    from .loader import NODE_DISPLAY_NAME_MAPPINGS as _loader_names
    from .conditioning import NODE_CLASS_MAPPINGS as _cond_maps
    from .conditioning import NODE_DISPLAY_NAME_MAPPINGS as _cond_names
    from .sampler import NODE_CLASS_MAPPINGS as _sampler_maps
    from .sampler import NODE_DISPLAY_NAME_MAPPINGS as _sampler_names
    from .vae import NODE_CLASS_MAPPINGS as _vae_maps
    from .vae import NODE_DISPLAY_NAME_MAPPINGS as _vae_names
    from .latent import NODE_CLASS_MAPPINGS as _latent_maps
    from .latent import NODE_DISPLAY_NAME_MAPPINGS as _latent_names
    from .memory import NODE_CLASS_MAPPINGS as _mem_maps
    from .memory import NODE_DISPLAY_NAME_MAPPINGS as _mem_names
    from .lora import NODE_CLASS_MAPPINGS as _lora_maps
    from .lora import NODE_DISPLAY_NAME_MAPPINGS as _lora_names
    from .controlnet import NODE_CLASS_MAPPINGS as _cn_maps
    from .controlnet import NODE_DISPLAY_NAME_MAPPINGS as _cn_names
    from .ip_adapter import NODE_CLASS_MAPPINGS as _ip_maps
    from .ip_adapter import NODE_DISPLAY_NAME_MAPPINGS as _ip_names

    # Optional: image chain, depth map, live preview
    try:
        from .image_chain import NODE_CLASS_MAPPINGS as _chain_maps
        from .image_chain import NODE_DISPLAY_NAME_MAPPINGS as _chain_names
    except Exception:
        _chain_maps = {}
        _chain_names = {}

    try:
        from .depth_map import NODE_CLASS_MAPPINGS as _depth_maps
        from .depth_map import NODE_DISPLAY_NAME_MAPPINGS as _depth_names
    except Exception:
        _depth_maps = {}
        _depth_names = {}

    try:
        from .live_preview import NODE_CLASS_MAPPINGS as _preview_maps
        from .live_preview import NODE_DISPLAY_NAME_MAPPINGS as _preview_names
    except Exception:
        _preview_maps = {}
        _preview_names = {}

    NODE_CLASS_MAPPINGS = {
        **_loader_maps,
        **_cond_maps,
        **_sampler_maps,
        **_vae_maps,
        **_latent_maps,
        **_mem_maps,
        **_lora_maps,
        **_cn_maps,
        **_ip_maps,
        **_chain_maps,
        **_depth_maps,
        **_preview_maps,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **_loader_names,
        **_cond_names,
        **_sampler_names,
        **_vae_names,
        **_latent_names,
        **_mem_names,
        **_lora_names,
        **_cn_names,
        **_ip_names,
        **_chain_names,
        **_depth_names,
        **_preview_names,
    }
except ModuleNotFoundError as exc:
    # Allow import on non-macOS / non-MLX hosts (e.g. CI, Comfy Registry parser)
    _missing = {
        "mlx", "numpy", "torch", "PIL", "safetensors", "gguf",
        "huggingface_hub", "transformers", "comfy", "folder_paths",
    }
    if exc.name in _missing:
        # Provide stub nodes that error at runtime with a helpful message
        class _Unavailable:
            CATEGORY = "ASDX/Utilities"
            RETURN_TYPES = ()
            FUNCTION = "_unavailable"
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}
            def _unavailable(self):
                raise RuntimeError(
                    "ASDX requires Apple Silicon MLX runtime dependencies. "
                    "Install on macOS with: pip install mlx mlx-lm safetensors"
                )

        NODE_CLASS_MAPPINGS = {
            name: type(name, (_Unavailable,), {"__doc__": None})
            for name in _DISPLAY
        }
        NODE_DISPLAY_NAME_MAPPINGS = dict(_DISPLAY)
    else:
        raise

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "_WEB_DIRECTORY",
    "__version__",
]
