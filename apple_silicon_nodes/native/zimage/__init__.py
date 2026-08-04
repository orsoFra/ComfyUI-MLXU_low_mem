"""
Z-Image (NextDiT / Lumina2 family) native MLX implementation.

Provides a complete MLX implementation of the Z-Image transformer:
- ZImageConfig: model configuration (dim=3840, 30 layers, full MHA)
- NextDiT: full transformer (context_refiner + noise_refiner + joint layers)
- Weight loading and mapping from Z-Image checkpoints

Usage:
    from apple_silicon_nodes.native.zimage import ZImageConfig, load_zimage_transformer
    transformer = load_zimage_transformer("z_image_bf16.safetensors", dtype="float16")
"""

from __future__ import annotations

from .config import ZImageConfig
from .model import (
    FeedForward,
    FinalLayer,
    JointAttention,
    JointTransformerBlock,
    NextDiT,
    embed_nd,
    load_zimage_transformer,
    timestep_embedding,
)
from .weight_map import map_zimage_to_native, normalize_zimage_keys

__all__ = [
    # Config
    "ZImageConfig",
    # Model
    "NextDiT",
    "JointAttention",
    "JointTransformerBlock",
    "FeedForward",
    "FinalLayer",
    "embed_nd",
    "timestep_embedding",
    "load_zimage_transformer",
    # Weight mapping
    "normalize_zimage_keys",
    "map_zimage_to_native",
]
