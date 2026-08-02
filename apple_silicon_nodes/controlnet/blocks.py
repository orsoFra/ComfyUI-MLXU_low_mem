"""ControlNet Union building blocks.

MLX-native layers used by ControlNetUnionModel:
  - ControlNetCondEmbedding: embed control image into conditioning features
  - TimeEmbedding: timestep embedding with optional text time augmentation
  - SinusoidalPositionalEncoding: sinusoidal positional encoding for timesteps
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class ControlNetCondEmbedding(nn.Module):
    """Embed control image into conditioning features."""

    def __init__(self, conditioning_channels: int = 3,
                 block_out_channels: list[int] | None = None,
                 embedding_channels: int = 320):
        super().__init__()
        if block_out_channels is None:
            block_out_channels = [32, 64, 128, 256]
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0],
                                 kernel_size=3, padding=1)
        self.blocks: list[nn.Module] = []
        for i in range(len(block_out_channels) - 1):
            ch_in = block_out_channels[i]
            ch_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(ch_in, ch_out, kernel_size=3,
                                         stride=2, padding=1))
        self.conv_out = nn.Conv2d(block_out_channels[-1], embedding_channels,
                                  kernel_size=3, padding=1)

    def __call__(self, conditioning: mx.array) -> mx.array:
        x = nn.silu(self.conv_in(conditioning))
        for block in self.blocks:
            x = nn.silu(block(x))
        return self.conv_out(x)


class TimeEmbedding(nn.Module):
    """Timestep embedding with optional text time augmentation."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(nn.silu(x))


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestep embeddings."""

    def __init__(self, dim: int, max_freq: float = 1, min_freq: float = 0,
                 scale: float = 1.0, cos_first: bool = True, full_turns: bool = True):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.scale = scale
        self.cos_first = cos_first
        self.full_turns = full_turns

    def __call__(self, t: mx.array) -> mx.array:
        """Compute sinusoidal encoding for timestep array."""
        t = t.astype(mx.float32)
        half_dim = self.dim // 2
        exponent = -math.log(10000) * mx.arange(start=0, stop=half_dim, dtype=mx.float32) / half_dim
        exponent = exponent + math.log(self.max_freq) - math.log(self.min_freq)
        emb = t[:, None] * mx.exp(exponent[None, :])
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
        if self.cos_first:
            emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
        return emb * self.scale
