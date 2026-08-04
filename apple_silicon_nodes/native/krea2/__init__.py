"""
Krea2 (SingleStreamDiT) native MLX implementation.

Provides a complete MLX implementation of the Krea2 model architecture:
- Krea2Config: model configuration (28 blocks, hidden=6144, GQA 48+12)
- SingleStreamDiT: full transformer with SingleStreamBlock
- RMSNorm: RMS normalization with (1 + scale) convention
- QKNorm: per-head Q/K normalization
- DoubleSharedModulation: timestep vec → 6 modulation params
- SimpleModulation: timestep vec → scale/shift
- Attention: GQA + QK norm + sigmoid gate
- SwiGLU: SwiGLU MLP
- TextFusionBlock/Transformer: Qwen3-VL layer fusion
- EmbedND: 3-axis RoPE embeddings
- Weight loading and mapping from Krea2 checkpoints

Usage:
    from apple_silicon_nodes.native.krea2 import (
        Krea2Config, SingleStreamDiT, load_krea2_transformer
    )
    config = Krea2Config(dtype="float16")
    transformer = load_krea2_transformer("krea2_raw.safetensors", dtype="float16")
"""

from __future__ import annotations

from .config import (
    Krea2Config,
    KREA2_LATENT_SCALE,
    KREA2_LATENT_SHIFT,
    process_krea2_latent_in,
    process_krea2_latent_out,
)
from .model import (
    Attention,
    DoubleSharedModulation,
    EmbedND,
    LastLayer,
    QKNorm,
    RMSNorm,
    SimpleModulation,
    SingleStreamBlock,
    SingleStreamDiT,
    SwiGLU,
    TextFusionBlock,
    TextFusionTransformer,
    load_krea2_transformer,
)
from .rope import apply_rope_3d, compute_rope_3d
from .weight_map import map_krea2_to_native, normalize_krea2_keys

__all__ = [
    # Config
    "Krea2Config",
    "KREA2_LATENT_SCALE",
    "KREA2_LATENT_SHIFT",
    "process_krea2_latent_in",
    "process_krea2_latent_out",
    # Model
    "SingleStreamDiT",
    "SingleStreamBlock",
    "Attention",
    "DoubleSharedModulation",
    "SimpleModulation",
    "RMSNorm",
    "QKNorm",
    "SwiGLU",
    "LastLayer",
    "TextFusionBlock",
    "TextFusionTransformer",
    "EmbedND",
    "load_krea2_transformer",
    # RoPE
    "apply_rope_3d",
    "compute_rope_3d",
    # Weight mapping
    "normalize_krea2_keys",
    "map_krea2_to_native",
]
