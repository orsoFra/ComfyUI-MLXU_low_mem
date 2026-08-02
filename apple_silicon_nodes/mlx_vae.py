"""
MLX VAE Decoder/Encoder
========================
Lightweight FLUX VAE implementation in MLX.

This provides a minimal VAE that can decode/encode FLUX latents.
For production use, integrate with the full sdmlx VAE implementation
which has proper weight loading from safetensors.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# Cache for loaded VAE modules
_vae_decoder: Optional[Any] = None
_vae_encoder: Optional[Any] = None


class GroupNorm(nn.Module):
    """Group normalization compatible with PyTorch GroupNorm."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = mx.zeros((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        B, C, H, W = x.shape
        x = x.reshape(B, self.num_groups, -1)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + self.eps
        x = (x - mean) / std
        x = x.reshape(B, C, H, W)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class ResBlock(nn.Module):
    """Residual block for the VAE."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class VAEDecoder(nn.Module):
    """FLUX VAE decoder in MLX.

    Simplified architecture matching FLUX's decoder structure.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.mid_block1 = ResBlock(channels * 4)
        self.mid_block2 = ResBlock(channels * 4)
        self.mid_block3 = ResBlock(channels * 4)

        # Upsampling blocks
        self.up_blocks = [
            _make_up_block(channels * 4, channels * 2),
            _make_up_block(channels * 2, channels),
        ]

        self.norm_out = GroupNorm(8, channels)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = self.mid_block3(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.conv_out(self.act(self.norm_out(x)))
        return x


class VAEEncoder(nn.Module):
    """FLUX VAE encoder in MLX (mirror of decoder)."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.mid_block1 = ResBlock(channels)
        self.mid_block2 = ResBlock(channels)
        self.mid_block3 = ResBlock(channels)

        # Downsampling blocks
        self.down_blocks = [
            _make_down_block(channels, channels * 2),
            _make_down_block(channels * 2, channels * 4),
        ]

        self.norm_out = GroupNorm(8, channels * 4)
        self.conv_out = nn.Conv2d(channels * 4, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = self.mid_block3(x)

        for block in self.down_blocks:
            x = block(x)

        x = self.conv_out(self.act(self.norm_out(x)))
        return x


def _make_up_block(channels_in: int, channels_out: int) -> nn.Sequential:
    return nn.Sequential(
        ResBlock(channels_in),
        nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
    )


def _make_down_block(channels_in: int, channels_out: int) -> nn.Sequential:
    return nn.Sequential(
        ResBlock(channels_in),
        nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1),
    )


def get_vae_decoder() -> VAEDecoder | None:
    """Get the cached VAE decoder instance."""
    global _vae_decoder
    if _vae_decoder is None:
        _vae_decoder = VAEDecoder()
    return _vae_decoder


def get_vae_encoder() -> VAEEncoder | None:
    """Get the cached VAE encoder instance."""
    global _vae_encoder
    if _vae_encoder is None:
        _vae_encoder = VAEEncoder()
    return _vae_encoder


def reset_vae() -> None:
    """Clear cached VAE instances."""
    global _vae_decoder, _vae_encoder
    _vae_decoder = None
    _vae_encoder = None
