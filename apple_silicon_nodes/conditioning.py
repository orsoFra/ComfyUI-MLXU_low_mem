"""
Conditioning nodes
==================
CLIP text encoding and conditioning manipulation for FLUX.

Nodes:
  - ASDX_DualCLIPLoader: Load CLIP-L + T5-XXL text encoders
  - ASDX_CLIPTextEncodeFlux: Encode text to T5 + CLIP embeddings
  - ASDX_ConditioningMerger: Merge two conditioning inputs
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch

import comfy.sd
import comfy.utils
from . import bridge


# ── Globals ───────────────────────────────────────────────────────────

_CLIP_CACHE: dict[str, Any] = {}


# ── Dual CLIP Loader ─────────────────────────────────────────────────

class ASDX_DualCLIPLoader:
    """Load CLIP-L and T5-XXL text encoders for FLUX.

    Returns an mlx_clip handle that can be used by the text encoder
    and sampler nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (cls._get_clip_names(),),
                "t5xxl_name": (cls._get_t5_names(),),
            },
        }

    RETURN_TYPES = ("mlx_clip",)
    RETURN_NAMES = ("mlx_clip",)
    FUNCTION = "load"
    CATEGORY = "ASDX/Loaders"

    @staticmethod
    def _get_clip_names() -> list[str]:
        try:
            import folder_paths
            return folder_paths.get_filename_list("text_encoders")
        except Exception:
            return ["clip_l.safetensors"]

    @staticmethod
    def _get_t5_names() -> list[str]:
        try:
            import folder_paths
            return folder_paths.get_filename_list("text_encoders")
        except Exception:
            return ["t5xxl.safetensors"]

    def load(self, clip_name: str, t5xxl_name: str) -> tuple[dict]:
        cache_key = f"{clip_name}:{t5xxl_name}"

        if cache_key not in _CLIP_CACHE:
            # Load the CLIP
            clip_path = self._find_file("text_encoders", clip_name)
            t5_path = self._find_file("text_encoders", t5xxl_name)

            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path, t5_path],
                embedding_directory=comfy.utils.get_t2ia_paths() if hasattr(comfy.utils, 'get_t2ia_paths') else [],
                clip_type=comfy.sd.CLIPType.FLUX,
            )

            _CLIP_CACHE[cache_key] = clip
            print(f"[ASDX] CLIP loaded: {clip_name} + {t5xxl_name}")

        return (_CLIP_CACHE[cache_key],)

    @staticmethod
    def _find_file(folder: str, name: str) -> str:
        try:
            import folder_paths
            return folder_paths.get_full_path(folder, name) or name
        except Exception:
            return name


# ── CLIP Text Encode ─────────────────────────────────────────────────

class ASDX_CLIPTextEncodeFlux:
    """Encode text prompts to FLUX conditioning (T5 + CLIP-L embeddings).

    Handles both CLIP-L and T5-XXL tokenization, returns mlx_conditioning
    that can be connected to the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mlx_clip": ("mlx_clip",),
                "clip_l": ("STRING", {"multiline": True, "default": ""}),
                "t5xxl": ("STRING", {"multiline": True, "default": ""}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "ASDX/Conditioning"

    def encode(self, mlx_clip: Any, clip_l: str, t5xxl: str, guidance: float) -> tuple[dict]:
        if not isinstance(mlx_clip, comfy.sd.CLIP):
            raise RuntimeError("ASDX: mlx_clip must be a Comfy CLIP object.")

        # Tokenize both text encoders
        tokens_l = mlx_clip.tokenize(clip_l)
        tokens_t5 = mlx_clip.tokenize(t5xxl)

        # Encode with scheduling
        conditioning = mlx_clip.encode_from_tokens_scheduled(
            {"l": tokens_l, "t5xxl": tokens_t5},
            add_dict={"guidance": float(guidance)},
        )

        result = {
            "type": "flux1",
            "conditioning": conditioning,
            "clip_l": clip_l,
            "t5xxl": t5xxl,
            "guidance": float(guidance),
        }

        print(f"[ASDX] Text encoded: clip_l={len(clip_l)} chars, t5xxl={len(t5xxl)} chars, "
              f"guidance={guidance:.1f}")

        return (result,)


# ── Conditioning Merger ──────────────────────────────────────────────

class ASDX_ConditioningMerger:
    """Merge two conditioning inputs into one.

    Useful for combining positive and negative conditioning, or for
    chaining multiple text encoders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING,mlx_conditioning",),
                "negative": ("CONDITIONING,mlx_conditioning",),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "merge"
    CATEGORY = "ASDX/Conditioning"

    def merge(self, positive: Any, negative: Any) -> tuple[dict]:
        """Merge conditioning - for FLUX, negative is typically ignored but accepted for compatibility."""
        # FLUX doesn't use negative conditioning in the traditional sense
        # Store both for compatibility but sampler will use positive
        result = dict(positive) if isinstance(positive, dict) else {
            "type": "flux1",
            "conditioning": positive,
        }
        result["_negative"] = negative
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ASDX_DualCLIPLoader": ASDX_DualCLIPLoader,
    "ASDX_CLIPTextEncodeFlux": ASDX_CLIPTextEncodeFlux,
    "ASDX_ConditioningMerger": ASDX_ConditioningMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_DualCLIPLoader": "🍏 ASDX Dual CLIP Loader",
    "ASDX_CLIPTextEncodeFlux": "🍏 ASDX CLIP Text Encode FLUX",
    "ASDX_ConditioningMerger": "🍏 ASDX Conditioning Merger",
}
