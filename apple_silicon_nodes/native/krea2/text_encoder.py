"""
Krea2 text encoder components — standalone TxtFusionTransformer.

This module provides the TextFusionTransformer as a standalone component
for use outside the main SingleStreamDiT model. In the full model, text
processing is integrated into the transformer's forward pass.

Architecture:
  Input:  [B, seq, txtlayers, txtdim]  (unpacked Qwen3-VL layer outputs)
  2 layerwise blocks (process each layer independently)
  Projector: Linear(txtlayers, 1) — fuses layer dimension
  2 refiner blocks (process combined text)
  Output: [B, seq, txtdim]
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .model import Attention, RMSNorm, SwiGLU


class TxtFusionBlock(nn.Module):
    """Single text fusion block.

    Architecture:
      x + attention(RMSNorm(x)) + SwiGLU(RMSNorm(x))
    """

    def __init__(self, dim: int, num_heads: int = 20, kv_heads: int | None = None):
        super().__init__()
        self.prenorm = RMSNorm(dim)
        self.postnorm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, kv_heads)
        self.mlp = SwiGLU(dim, multiplier=4)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.prenorm(x))
        x = x + self.mlp(self.postnorm(x))
        return x


class TxtFusionTransformer(nn.Module):
    """Text fusion adapter for Qwen3-VL layer taps.

    Fuses multiple Qwen3-VL layer outputs into a single-layer representation.

    Architecture:
      Input:  [B, seq, txtlayers, txtdim]
      2 layerwise blocks (process each layer independently)
      Rearrange: [B, seq, txtlayers, txtdim] → [B, seq, txtdim, txtlayers]
      Projector: Linear(txtlayers, 1) → [B, seq, txtdim, 1]
      2 refiner blocks
      Output: [B, seq, txtdim]
    """

    def __init__(self, num_txt_layers: int = 12, text_dim: int = 2560,
                 heads: int = 20, kv_heads: int | None = None):
        super().__init__()
        self.num_txt_layers = num_txt_layers
        self.text_dim = text_dim
        self.layerwise_blocks = [
            TxtFusionBlock(text_dim, heads, kv_heads) for _ in range(2)
        ]
        # Projector: Linear(num_txt_layers, 1) — fuses layer dimension
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = [
            TxtFusionBlock(text_dim, heads, kv_heads) for _ in range(2)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through text fusion transformer.

        Args:
            x: Unpacked Qwen3-VL outputs [B, seq, txtlayers, txtdim].

        Returns:
            Fused text embeddings [B, seq, txtdim].
        """
        B, seq, txtlayers, txtdim = x.shape

        # Process each layer independently through layerwise blocks
        x = x.reshape(B * seq, txtlayers, txtdim)
        for block in self.layerwise_blocks:
            x = block(x)

        # Rearrange: [B*seq, txtlayers, txtdim] → [B, seq, txtdim, txtlayers]
        x = x.reshape(B, seq, txtlayers, txtdim).transpose(0, 1, 3, 2)

        # Project: [B, seq, txtdim, txtlayers] → [B, seq, txtdim, 1]
        x = self.projector(x).squeeze(-1)

        # Refine combined text through refiner blocks
        for block in self.refiner_blocks:
            x = block(x)

        return x
