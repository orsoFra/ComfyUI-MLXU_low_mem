"""
Configuration module for the native MLX transformer.

Centralizes hyperparameters and provides validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class FluxConfig:
    """FLUX.1 model architecture configuration.

    Attributes:
        num_double_blocks: Number of double transformer blocks (19 for FLUX.1-dev)
        num_single_blocks: Number of single transformer blocks (38 for FLUX.1-dev)
        hidden_dim: Hidden dimension size (3072 for FLUX.1)
        mlp_dim: MLP expansion dimension (12288 for FLUX.1)
        num_heads: Number of attention heads (24 for FLUX.1)
        guidance_embed: Whether the model includes guidance embedding (dev=True, schnell=False)
        dtype: MLX dtype string ("float16" or "bfloat16")
    """
    num_double_blocks: int = 19
    num_single_blocks: int = 38
    hidden_dim: int = 3072
    mlp_dim: int = 12288
    num_heads: int = 24
    guidance_embed: bool = True
    dtype: str = "float16"

    @property
    def mlx_dtype(self) -> mx.Dtype:
        """Convert dtype string to mlx.core dtype."""
        dtype_map = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(f"ASDX: unsupported dtype '{self.dtype}'. Use float16, bfloat16, or float32.")
        return dtype_map[self.dtype]

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_double_blocks > 0, "num_double_blocks must be positive"
        assert self.num_single_blocks > 0, "num_single_blocks must be positive"
        assert self.dtype in ("float16", "bfloat16", "float32"), \
            f"unsupported dtype: {self.dtype}"

    def __post_init__(self) -> None:
        self.validate()


# ── FLUX latent space constants ──────────────────────────────────────────
# Adapted from DiffusionKit's FluxLatentFormat

FLUX_LATENT_SCALE: float = 0.3611
"""Scale factor for FLUX latent space transformation."""

FLUX_LATENT_SHIFT: float = 0.1159
"""Shift factor for FLUX latent space transformation."""


def process_flux_latent_in(latent: Any) -> Any:
    """Process latent for model input: (latent - shift) * scale.

    Args:
        latent: Input latent tensor (any array-like).

    Returns:
        Processed latent tensor.
    """
    return (latent - FLUX_LATENT_SHIFT) * FLUX_LATENT_SCALE


def process_flux_latent_out(latent: Any) -> Any:
    """Process latent for model output: latent / scale + shift.

    Args:
        latent: Output latent tensor (any array-like).

    Returns:
        Reconstructed latent tensor.
    """
    return (latent / FLUX_LATENT_SCALE) + FLUX_LATENT_SHIFT
