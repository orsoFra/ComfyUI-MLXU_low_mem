"""
Flux2 (FLUX.2/Klein) native MLX implementation.

Provides a complete MLX implementation of the Flux2/Klein transformer:
- Flux2Config: model configuration (hidden_size=4096, 8 double + 24 single
  blocks for the 9B Klein checkpoint verified on this machine)
- Flux2Transformer: full transformer with global modulation + SiLU-gated MLP
- Weight loading and mapping from Flux2/Klein checkpoints

Usage:
    from apple_silicon_nodes.native.flux2 import Flux2Config, load_flux2_transformer
    transformer = load_flux2_transformer("flux2Klein_9b.safetensors", dtype="float16")
"""

from __future__ import annotations

from .config import (
    Flux2Config,
    FLUX2_LATENT_SCALE,
    FLUX2_LATENT_SHIFT,
    detect_flux2_config,
    process_flux2_latent_in,
    process_flux2_latent_out,
)
from .model import (
    DoubleBlock,
    Flux2Transformer,
    LastLayer,
    MLPEmbedder,
    Modulation,
    QKNorm,
    SelfAttentionProj,
    SingleBlock,
    embed_nd,
    load_flux2_transformer,
    timestep_embedding,
)
from .weight_map import map_flux2_to_native, normalize_flux2_keys

__all__ = [
    # Config
    "Flux2Config",
    "FLUX2_LATENT_SCALE",
    "FLUX2_LATENT_SHIFT",
    "detect_flux2_config",
    "process_flux2_latent_in",
    "process_flux2_latent_out",
    # Model
    "Flux2Transformer",
    "DoubleBlock",
    "SingleBlock",
    "LastLayer",
    "MLPEmbedder",
    "Modulation",
    "QKNorm",
    "SelfAttentionProj",
    "embed_nd",
    "timestep_embedding",
    "load_flux2_transformer",
    # Weight mapping
    "normalize_flux2_keys",
    "map_flux2_to_native",
]
