"""
SDXL (UNetModel) native MLX implementation.

Provides a complete MLX implementation of the SDXL UNet architecture:
- SDXLConfig: model configuration (320 base channels, 3 resolution levels)
- UNetModel: full conv UNet (ResBlock + SpatialTransformer, ADM conditioning)
- encode_adm: builds the ADM/"y" vector (pooled CLIP-G + size/crop embeddings)
- Weight loading and mapping from SDXL/Illustrious/Pony checkpoints

Usage:
    from apple_silicon_nodes.native.sdxl import SDXLConfig, load_sdxl_unet
    unet = load_sdxl_unet("sd_xl_base_1.0.safetensors", dtype="float16")
"""

from __future__ import annotations

from .config import (
    SDXLConfig,
    SDXL_LATENT_SCALE,
    process_sdxl_latent_in,
    process_sdxl_latent_out,
)
from .model import (
    BasicTransformerBlock,
    CrossAttention,
    Downsample,
    GEGLU,
    ResBlock,
    SpatialTransformer,
    UNetModel,
    Upsample,
    encode_adm,
    load_sdxl_unet,
    timestep_embedding,
)
from .weight_map import map_sdxl_to_native, normalize_sdxl_keys

__all__ = [
    # Config
    "SDXLConfig",
    "SDXL_LATENT_SCALE",
    "process_sdxl_latent_in",
    "process_sdxl_latent_out",
    # Model
    "UNetModel",
    "ResBlock",
    "CrossAttention",
    "BasicTransformerBlock",
    "SpatialTransformer",
    "Downsample",
    "Upsample",
    "GEGLU",
    "encode_adm",
    "timestep_embedding",
    "load_sdxl_unet",
    # Weight mapping
    "normalize_sdxl_keys",
    "map_sdxl_to_native",
]
