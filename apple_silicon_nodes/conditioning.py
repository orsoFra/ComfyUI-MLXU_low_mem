"""
Conditioning nodes
==================
CLIP text encoding and conditioning manipulation for FLUX and SD-style models.

Nodes:
  - ASDX_DualCLIPLoader: Load two CLIP text encoders (SDXL, FLUX, SD3...)
  - ASDX_CLIPLoader: Load a single CLIP model (SD1.5, Pony, Illustrious...)
  - ASDX_CLIPTextEncode: Encode text to conditioning (auto-detect FLUX/SD mode)
  - ASDX_ConditioningMerger: Merge two conditioning inputs
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch

import comfy.sd
import comfy.text_encoders.krea2
import comfy.utils
from . import bridge


# ── Globals ───────────────────────────────────────────────────────────

_CLIP_CACHE: dict[str, Any] = {}

# Krea2 Identity Edit grounded-encode template: the trained conditioning
# template (`comfy/text_encoders/krea2.py::KREA2_TEMPLATE`) with a vision
# block inserted before the user text -- ported from comfyui-krea2edit's
# `Krea2EditGroundedEncode` (Apache-2.0). Required because Krea2Tokenizer only
# overrides the *no-image* template; passing `images=` alone falls back to the
# base Qwen3VLTokenizer's generic vision template (no system prompt), silently
# mismatching what the krea2_edit LoRA was trained on.
# `_KREA2_DEFAULT_GROUNDING_SYSTEM` is the training-default system prompt --
# generic ("objects and background", no explicit person/face wording).
# `system_prompt` on the node overrides it, e.g. to steer the vision encoder
# toward facial identity detail, matching the reference's own override input.
_KREA2_DEFAULT_GROUNDING_SYSTEM = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)


def _krea2_grounding_template(system_prompt: str) -> str:
    sp = system_prompt.strip() or _KREA2_DEFAULT_GROUNDING_SYSTEM
    return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n")


def _clip_model_options(type_str: str) -> dict:
    """comfy's default text-encoder dtype is float16 (`model_management.
    text_encoder_dtype`), even on CPU. That's fine for text-only encoding, but
    Krea2's Qwen3-VL vision tower only actually runs when grounded (image
    input on ASDX_CLIPTextEncode), and float16 vision-transformer attention
    overflows far more easily than bf16 (same CLAUDE.md caveat as MPS
    LayerNorm/VAE) -- observed producing NaN embeddings that decode to a black
    image. bf16 is the same 2 bytes/param as fp16, so this costs nothing.
    """
    if type_str == "krea2":
        return {"dtype": torch.bfloat16}
    return {}


# ── Dual CLIP Loader ─────────────────────────────────────────────────

class ASDX_DualCLIPLoader:
    """Load two CLIP text encoders for dual-CLIP architectures.

    Supports SDXL, SD3, FLUX, Hunyuan, HiDream, Kandinsky, LTXV, Newbie, ACE.
    Returns an mlx_clip handle that can be used by the text encoder
    and sampler nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name1": (cls._get_clip_names(),),
                "clip_name2": (cls._get_clip_names(),),
                "type": (_DUAL_CLIP_TYPES, {"default": "flux"}),
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

    def load(self, clip_name1: str, clip_name2: str, type: str) -> tuple[dict]:
        cache_key = f"{clip_name1}:{clip_name2}:{type}"

        if cache_key not in _CLIP_CACHE:
            # Only one CLIP pair is meaningfully "current" at a time -- evict
            # prior entries before loading a new one instead of accumulating
            # every distinct clip_name/type combo used in the session (same
            # fix already applied to loader.py's _MODEL_CACHE).
            if _CLIP_CACHE:
                _CLIP_CACHE.clear()
                bridge.clear_mlx_cache()

            # Load the CLIP
            clip_path1 = self._find_file("text_encoders", clip_name1)
            clip_path2 = self._find_file("text_encoders", clip_name2)
            clip_type_enum = _clip_type_from_string(type)

            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
                embedding_directory=comfy.utils.get_t2ia_paths() if hasattr(comfy.utils, 'get_t2ia_paths') else [],
                clip_type=clip_type_enum,
                model_options=_clip_model_options(type),
            )

            _CLIP_CACHE[cache_key] = clip
            print(f"[ASDX] Dual CLIP loaded: {clip_name1} + {clip_name2} (type={type})")

        return (_CLIP_CACHE[cache_key],)

    @staticmethod
    def _find_file(folder: str, name: str) -> str:
        try:
            import folder_paths
            return folder_paths.get_full_path(folder, name) or name
        except Exception:
            return name


# ── CLIP Text Encode ─────────────────────────────────────────────────

class ASDX_CLIPTextEncode:
    """Encode text prompts to conditioning for any model type.

    Auto-detects FLUX vs SD-style encoding:
    - If t5xxl is provided → FLUX mode (separate clip_l + t5xxl + guidance)
    - Otherwise → SD/SDXL/Pony mode (single CLIP encode)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mlx_clip": ("mlx_clip",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "t5xxl": ("STRING", {"multiline": True, "default": ""}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "image": ("IMAGE", {"tooltip": "Krea2 Identity Edit: source image, encoded "
                                                "through the CLIP's vision tower alongside the "
                                                "prompt (image-grounded conditioning). Ignored "
                                                "for non-Krea2 CLIPs."}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                          "tooltip": "cap longest side fed to the vision tower; "
                                                     "0 = native resolution"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                              "tooltip": "advanced (optional): override the "
                                                         "grounding system prompt (empty = "
                                                         "training default). Steers what the "
                                                         "vision encoder attends to, e.g. "
                                                         "facial identity detail."}),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "ASDX/Conditioning"

    @staticmethod
    def _prep_grounding_image(image: torch.Tensor, grounding_px: int) -> torch.Tensor:
        """Resize an IMAGE for the vision tower, matching comfyui-krea2edit's
        Krea2EditGroundedEncode._prep -- the krea2_edit LoRA was trained with
        384-768px jitter, so capping here keeps inference in-distribution.
        """
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(
        self,
        mlx_clip: Any,
        text: str,
        t5xxl: str = "",
        guidance: float = 3.5,
        image: torch.Tensor | None = None,
        grounding_px: int = 768,
        system_prompt: str = "",
    ) -> tuple[dict]:
        if not isinstance(mlx_clip, comfy.sd.CLIP):
            raise RuntimeError("ASDX: mlx_clip must be a Comfy CLIP object.")

        if t5xxl:
            # FLUX mode: separate clip_l + t5xxl inputs. mlx_clip.tokenize(text)
            # already returns the full {"l": [...], "t5xxl": [...]} dict (a
            # dual-tokenizer CLIP tokenizes through every sub-tokenizer at
            # once) -- only the "t5xxl" entry needs overriding with its own
            # text, matching comfy's own CLIPTextEncodeFlux.execute(). Wrapping
            # both full dicts again as {"l": tokens_l, "t5xxl": tokens_t5}
            # (the previous code here) nests them one level too deep, so
            # encode_token_weights() ends up iterating dict keys ("l",
            # "t5xxl") as if they were (token, weight) pairs.
            tokens = mlx_clip.tokenize(text)
            tokens["t5xxl"] = mlx_clip.tokenize(t5xxl)["t5xxl"]
            conditioning = mlx_clip.encode_from_tokens_scheduled(
                tokens,
                add_dict={"guidance": float(guidance)},
            )
            result = {
                "type": "flux1",
                "conditioning": conditioning,
                "text": text,
                "t5xxl": t5xxl,
                "guidance": float(guidance),
            }
            print(f"[ASDX] Text encoded (FLUX): text={len(text)} chars, "
                  f"t5xxl={len(t5xxl)} chars, guidance={guidance:.1f}")
        else:
            # SD / SDXL / Pony mode: single CLIP encode
            tokenize_kwargs = {}
            grounded = False
            if image is not None:
                if isinstance(mlx_clip.tokenizer, comfy.text_encoders.krea2.Krea2Tokenizer):
                    tokenize_kwargs["images"] = [self._prep_grounding_image(image, grounding_px)]
                    tokenize_kwargs["llama_template"] = _krea2_grounding_template(system_prompt)
                    grounded = True
                else:
                    print("[ASDX] Warning: 'image' input ignored -- image-grounded encoding "
                          "is only implemented for Krea2 CLIPs.")
            tokens = mlx_clip.tokenize(text, **tokenize_kwargs)
            conditioning = mlx_clip.encode_from_tokens_scheduled(tokens)
            # Fail fast on a corrupt (NaN/Inf) embedding rather than letting it
            # silently ride through ~7min of diffusion sampling and VAE decode
            # to surface only as a black output image (comfy's own PIL cast then
            # warns "invalid value encountered in cast" and clips to garbage).
            # Seen in practice: the grounded (image-conditioned) path exercises
            # the vision tower for the first time, and comfy's default text
            # encoder dtype is float16 (`model_management.text_encoder_dtype`),
            # which overflows more easily than bf16 in vision-transformer
            # attention -- see the dtype override below.
            for cond, _ in conditioning:
                if not torch.isfinite(cond).all():
                    raise RuntimeError(
                        "ASDX: CLIP Text Encode produced a non-finite (NaN/Inf) "
                        "embedding" + (" while image-grounded" if grounded else "") +
                        " -- aborting before the expensive sampling pass."
                    )
            result = {
                "type": "clip",
                "conditioning": conditioning,
                "text": text,
            }
            print(f"[ASDX] Text encoded (SD-style): {len(text)} chars, type=clip"
                  f"{', grounded on source image' if grounded else ''}")

        return (result,)


# ── CLIP Types ────────────────────────────────────────────────────────
# Complete mapping of ComfyUI's CLIPType enum to human-readable strings.
# Single types (CLIPLoader node): all 33 CLIPType values + "mage" alias
# Dual types (DualCLIPLoader node): 9 types for two-CLIP architectures

_SINGLE_CLIP_TYPES: list[str] = [
    "stable_diffusion",   # SD1.5 — clip-l
    "stable_cascade",     # Stable Cascade — clip-g
    "sd3",                # SD3 — clip-g + clip-l + t5
    "stable_audio",       # Stable Audio — t5 base
    "hunyuan_dit",        # Hunyuan DiT
    "flux",               # FLUX — clip-l + t5
    "mochi",              # Mochi — t5 xxl
    "ltxv",               # LTX-Video
    "hunyuan_video",      # Hunyuan Video
    "pixart",             # PixArt — gemma 2 2B
    "cosmos",             # Cosmos — old t5 xxl
    "lumina2",            # Lumina2 — gemma 2 2B
    "wan",                # Wan — umt5 xxl
    "hidream",            # HiDream — t5 + llama
    "chroma",             # Chroma
    "ace",                # ACE
    "omnigen2",           # OmniGen2 — qwen vl 2.5 3B
    "qwen_image",         # Qwen Image
    "hunyuan_image",      # Hunyuan Image — qwen2.5vl + byt5
    "hunyuan_video_15",   # Hunyuan Video 1.5
    "ovis",               # OVIS
    "kandinsky5",         # Kandinsky 5
    "kandinsky5_image",   # Kandinsky 5 Image
    "newbie",             # Newbie — gemma-3-4b-it + jina
    "flux2",              # FLUX 2
    "longcat_image",      # LongCat Image
    "cogvideox",          # CogVideoX — t5 xxl (226-token)
    "lens",               # Lens — gpt-oss-20b
    "pixeldit",           # PixelDIT — gemma 2 2B elm
    "ideogram4",          # Ideogram 4
    "boogu",              # Boogu
    "krea2",              # Krea2
    "joyimage",           # JoyImage — qwen3-vl 8B
    "mage",               # Mage (alias → stable_diffusion)
]

_DUAL_CLIP_TYPES: list[str] = [
    "sdxl",               # clip-l + clip-g
    "sd3",                # clip-l + clip-g / clip-l + t5 / clip-g + t5
    "flux",               # clip-l + t5
    "hunyuan_video",      # hunyuan video dual
    "hidream",            # t5 + llama
    "hunyuan_image",      # qwen2.5vl + byt5
    "hunyuan_video_15",   # hunyuan video 1.5 dual
    "kandinsky5",         # kandinsky 5 dual
    "kandinsky5_image",   # kandinsky 5 image dual
    "ltxv",               # ltxv dual
    "newbie",             # gemma-3-4b-it + jina
    "ace",                # ace dual
]


def _clip_type_from_string(s: str) -> Any:
    """Convert a string name to the corresponding CLIPType enum value.

    Matches ComfyUI's CLIPLoader.load_clip() behavior: getattr(CLIPType, name.upper()).
    Falls back to STABLE_DIFFUSION for unknown types.
    """
    # Map special aliases
    alias_map = {
        "mage": "STABLE_DIFFUSION",
        "sd1": "SD15",
        "sd2": "SD21",
        "flux_hybrid": "HYBRID",
        "pony": "PONY",
    }
    key = alias_map.get(s, s.upper())
    return getattr(comfy.sd.CLIPType, key, comfy.sd.CLIPType.STABLE_DIFFUSION)


class ASDX_CLIPLoader:
    """Load a single CLIP text encoder model.

    Supports all ComfyUI CLIP types: SD1.5, SDXL, Pony, SD3, FLUX, FLUX2,
    Hunyuan, Mochi, Wan, PixArt, Kandinsky, Krea2, JoyImage, and more.
    Returns an mlx_clip handle that can be connected to ASDX_CLIPTextEncode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (cls._get_clip_names(),),
                "type": (_SINGLE_CLIP_TYPES, {"default": "stable_diffusion"}),
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

    def load(self, clip_name: str, type: str) -> tuple[dict]:
        cache_key = f"{clip_name}:{type}"

        if cache_key not in _CLIP_CACHE:
            # See ASDX_DualCLIPLoader.load for why prior entries are evicted
            # here (shared _CLIP_CACHE, same one-active-entry policy as
            # loader.py's _MODEL_CACHE).
            if _CLIP_CACHE:
                _CLIP_CACHE.clear()
                bridge.clear_mlx_cache()

            clip_path = self._find_file("text_encoders", clip_name)
            clip_type_enum = _clip_type_from_string(type)

            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=comfy.utils.get_t2ia_paths() if hasattr(comfy.utils, 'get_t2ia_paths') else [],
                clip_type=clip_type_enum,
                model_options=_clip_model_options(type),
            )

            _CLIP_CACHE[cache_key] = clip
            print(f"[ASDX] CLIP loaded: {clip_name} (type={type})")

        return (_CLIP_CACHE[cache_key],)

    @staticmethod
    def _find_file(folder: str, name: str) -> str:
        try:
            import folder_paths
            return folder_paths.get_full_path(folder, name) or name
        except Exception:
            return name


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
    "ASDX_CLIPLoader": ASDX_CLIPLoader,
    "ASDX_CLIPTextEncode": ASDX_CLIPTextEncode,
    "ASDX_ConditioningMerger": ASDX_ConditioningMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_DualCLIPLoader": "🍏 ASDX Dual CLIP Loader",
    "ASDX_CLIPLoader": "🍏 ASDX CLIP Loader",
    "ASDX_CLIPTextEncode": "🍏 ASDX CLIP Text Encode",
    "ASDX_ConditioningMerger": "🍏 ASDX Conditioning Merger",
}
