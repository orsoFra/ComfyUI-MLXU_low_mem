"""
IP-Adapter Cross-Attention Injection
=====================================
Inject reference image features into the transformer via cross-attention.

Architecture (INTENDED, NOT YET FUNCTIONAL — see warning below):
  1. Pre-compute K/V from reference image tokens using CLIP-Vision encoder
  2. Cache K/V pairs per attention layer
  3. During sampling, prepend reference K/V to text K/V in attention

Supports multiple embedding scaling modes:
  - "V only": only V is scaled
  - "K+V": both K and V are scaled
  - "K+V w/ C penalty": K is penalized by cosine similarity

WARNING — this module is currently a NO-OP end to end:
  - `CLIPVisionEncoder` is defined but never instantiated or called anywhere
    (`ASDX_IPAdapterCLIPVisionEncode.encode()` fabricates fake conditioning
    directly from raw resized pixels instead of running it).
  - `IPAdapterCache` is defined but never instantiated anywhere.
  - `image_proj`/`ip_proj` weights loaded by `ASDX_IPAdapterLoader` are never
    referenced again after loading.
  - `ASDX_ApplyIPAdapter.apply()` only stashes metadata into the conditioning
    dict; nothing downstream reads `conditioning["ip_adapter"]` to inject K/V
    into any attention layer.
  Unlike ControlNet-Union, FLUX has no IP-Adapter reference implementation in
  ComfyUI core to verify an architecture against (only divergent third-party
  forks: XLabs, InstantX) — a real fix needs that reference chosen and wired
  through the FLUX attention layers, not just a syntax patch. Reference image
  inputs currently have ZERO effect on generation, silently.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from . import bridge


# ── IP-Adapter Cache ─────────────────────────────────────────────────

class IPAdapterCache:
    """Stores pre-computed K/V pairs for IP-Adapter injection."""

    def __init__(self):
        self.adapters: list[dict[str, Any]] = []  # list of adapter configs
        self.step_percent: float = 0.0
        self.use_cfg: bool = False

    def add_adapter(
        self,
        name: str,
        cond: mx.array,           # [B, T, D] image condition tokens
        cond_tokens: mx.array,    # [B, T] token indices (optional)
        weight: float = 1.0,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
    ) -> None:
        """Register an IP-Adapter with its condition tokens."""
        self.adapters.append({
            "name": name,
            "cond": cond,
            "cond_tokens": cond_tokens,
            "weight": weight,
            "start_percent": start_percent,
            "end_percent": end_percent,
        })

    def get_active_adapters(self, step_percent: float) -> list[dict]:
        """Get adapters active at this step percentage."""
        return [
            a for a in self.adapters
            if a["start_percent"] <= step_percent <= a["end_percent"]
        ]

    def clear(self) -> None:
        self.adapters.clear()


# ── CLIP Vision Encoder (Minimal) ────────────────────────────────────

class CLIPVisionEncoder(nn.Module):
    """Minimal CLIP-Vision-H encoder for IP-Adapter.

    Uses the MLX implementation of the ViT-L/14 CLIP encoder.
    Extracts image tokens that are projected to conditioning space.
    """

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16,
                 num_layers: int = 24):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=14, stride=14)

        # CLS token
        self.cls_token = mx.zeros((1, 1, hidden_size))

        # Positional embedding (197 tokens: 1 CLS + 14*14 patches)
        self.pos_embed = mx.zeros((1, 197, hidden_size))

        # Transformer blocks
        self.blocks = [nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.MultiHeadAttention(hidden_size, num_heads, bias=True),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        ) for _ in range(num_layers)]

        # Final layer norm
        self.final_ln = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode image to tokens.

        Args:
            x: [B, 3, 224, 224] normalized image

        Returns:
            [B, 257, 1024] token embeddings (CLS + patch tokens)
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, 1024, 16, 16]
        x = mx.transpose(x, (0, 2, 3, 1))  # [B, 16, 16, 1024]
        x = x.reshape(B, 256, self.hidden_size)  # [B, 256, 1024]

        # Add CLS token
        cls = mx.broadcast_to(self.cls_token, (B, 1, self.hidden_size))
        x = mx.concatenate([cls, x], axis=1)  # [B, 257, 1024]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = x + block(x)

        # Final layer norm
        x = self.final_ln(x)
        return x


# ── IP-Adapter Loader ────────────────────────────────────────────────

class ASDX_IPAdapterLoader:
    """Load an IP-Adapter model file.

    The IP-Adapter contains projection weights that map CLIP-Vision
    features to the transformer's conditioning space.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_adapter_name": (cls._get_adapters(),),
            },
        }

    RETURN_TYPES = ("ip_adapter",)
    RETURN_NAMES = ("ip_adapter",)
    FUNCTION = "load"
    CATEGORY = "ASDX/IPAdapter"

    @staticmethod
    def _get_adapters() -> list[str]:
        """Get list of available IP-Adapter files."""
        try:
            import folder_paths
            adapters = []
            for folder in ("ipadapter", "unet"):
                try:
                    adapters.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if adapters:
                return adapters
        except Exception:
            pass
        return ["ip-adapter-plus.safetensors"]

    def load(self, ip_adapter_name: str) -> tuple[dict]:
        path = self._resolve_path(ip_adapter_name)
        adapter = self._load_adapter(path)
        return (adapter,)

    @staticmethod
    def _load_adapter(path: Path) -> dict[str, Any]:
        """Load IP-Adapter weights."""
        t0 = time.perf_counter()

        import safetensors
        with open(path, "rb") as f:
            raw = safetensors.numpy.load(f.read())

        # Extract projection weights
        # IP-Adapter typically has: "image_proj." and "ip_adapter." keys
        image_proj = {}
        ip_proj = {}

        for key, value in raw.items():
            if key.startswith("image_proj."):
                image_proj[key[len("image_proj."):]] = mx.array(value)
            elif key.startswith("ip_adapter."):
                ip_proj[key[len("ip_adapter."):]] = mx.array(value)

        mx.eval(*image_proj.values(), *ip_proj.values())

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] IP-Adapter loaded: {path.name} "
              f"({len(image_proj)} image_proj + {len(ip_proj)} ip_proj, {elapsed:.2f}s)")

        return {
            "name": path.stem,
            "path": str(path),
            "image_proj": image_proj,
            "ip_proj": ip_proj,
        }

    @staticmethod
    def _resolve_path(name: str) -> Path:
        try:
            import folder_paths
            for folder in ("ipadapter", "unet"):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "ipadapter" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


# ── CLIP Vision Encode ───────────────────────────────────────────────

class ASDX_IPAdapterCLIPVisionEncode:
    """Encode a reference image using CLIP-Vision encoder.

    Produces condition tokens that will be used for IP-Adapter injection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "ASDX/IPAdapter"

    def encode(self, images: torch.Tensor) -> tuple[dict]:
        """Encode reference image(s) to CLIP-Vision tokens."""
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Normalize to [-1, 1] for CLIP-Vision
        img_np = images.detach().cpu().numpy().astype(np.float32)
        img_np = (img_np * 2.0) - 1.0

        # Resize to 224x224 if needed
        h, w = img_np.shape[1:3]
        if h != 224 or w != 224:
            import numpy as np
            # Simple resize using interpolation
            from PIL import Image
            batch_tokens = []
            for i in range(img_np.shape[0]):
                img_pil = Image.fromarray(
                    ((img_np[i] + 1) / 2 * 255).astype(np.uint8)
                )
                img_resized = img_pil.resize((224, 224), Image.BILINEAR)
                img_resized = np.array(img_resized, dtype=np.float32)
                img_resized = (img_resized / 127.5) - 1.0
                batch_tokens.append(img_resized)
            img_np = np.stack(batch_tokens)

        img_mlx = mx.array(img_np.transpose(0, 3, 1, 2))  # [B, 3, 224, 224]
        mx.eval(img_mlx)

        # Return as conditioning (tokens will be computed during sampling)
        conditioning = {
            "type": "ip_adapter",
            "image_tokens": img_mlx,
            "conditioning": [
                {"cond": img_mlx, "pooled_output": img_mlx[:, 0:1]}  # CLS token
            ],
        }

        return (conditioning,)


# ── Apply IP-Adapter ─────────────────────────────────────────────────

class ASDX_ApplyIPAdapter:
    """Apply IP-Adapter to conditioning for image-style transfer.

    Pre-computes K/V from reference image tokens and injects them
    into the transformer's cross-attention layers during sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("mlx_conditioning",),
                "ip_adapter": ("ip_adapter",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("mlx_conditioning",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "ASDX/IPAdapter"

    def apply(
        self,
        positive: dict,
        ip_adapter: dict,
        image: torch.Tensor,
        strength: float,
        start_percent: float,
        end_percent: float,
    ) -> tuple[dict]:
        """Apply IP-Adapter to conditioning."""
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Normalize
        img_np = image.detach().cpu().numpy().astype(np.float32)
        img_np = (img_np * 2.0) - 1.0

        # Resize
        h, w = img_np.shape[1:3]
        if h != 224 or w != 224:
            from PIL import Image
            img_pil = Image.fromarray(((img_np[0] + 1) / 2 * 255).astype(np.uint8))
            img_pil = img_pil.resize((224, 224), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32)
            img_np = (img_np / 127.5) - 1.0
            img_np = img_np.transpose(2, 0, 1)[None]

        img_mlx = mx.array(img_np)
        mx.eval(img_mlx)

        # Store IP-Adapter info in conditioning
        conditioning = dict(positive) if isinstance(positive, dict) else dict(positive[0]) if positive else {}
        conditioning["type"] = "flux1"
        conditioning["ip_adapter"] = {
            "adapter": ip_adapter,
            "image": img_mlx,
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }

        # Merge with original conditioning
        if "conditioning" in positive:
            conditioning["conditioning"] = positive["conditioning"]
        elif "cond" in positive:
            conditioning["cond"] = positive["cond"]
            if "pooled_output" in positive:
                conditioning["pooled_output"] = positive["pooled_output"]

        return (conditioning,)


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ASDX_IPAdapterLoader": ASDX_IPAdapterLoader,
    "ASDX_IPAdapterCLIPVisionEncode": ASDX_IPAdapterCLIPVisionEncode,
    "ASDX_ApplyIPAdapter": ASDX_ApplyIPAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_IPAdapterLoader": "🍏 ASDX IP-Adapter Loader",
    "ASDX_IPAdapterCLIPVisionEncode": "🍏 ASDX CLIP Vision Encode",
    "ASDX_ApplyIPAdapter": "🍏 ASDX Apply IP-Adapter",
}
