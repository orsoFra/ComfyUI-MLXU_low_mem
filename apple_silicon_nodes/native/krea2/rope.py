"""
3-axis Rotary Positional Embeddings (RoPE) for Krea2 Identity Edit.

Krea2 uses a 3-axis RoPE scheme (frame, height, width) instead of FLUX's
2-axis scheme (index, height, width). This enables the Identity Edit
capability where source images are prepended as frame=1 tokens.

Axes dimensions: [32, 48, 48] summing to hidden_dim=6144 (head_dim=128).
Theta: 1000 (vs FLUX's 10000).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _rope_axis(pos: mx.array, dim: int, theta: float) -> tuple[mx.array, mx.array]:
    """Compute cosine and sine frequencies for a single RoPE axis.

    Args:
        pos: Position indices [B, N] or [N].
        dim: Dimension for this axis.
        theta: Base frequency.

    Returns:
        (cos, sin) tensors of shape [B, N, dim] or [N, dim].
    """
    # freqs = theta^(-2i/d) for i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    # pos: [B, N] or [N], inv_freq: [dim/2]
    # Result: angles [B, N, dim/2]
    angles = pos[..., None] * inv_freq[None, None, :]
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    return cos, sin


def apply_rope_3d(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    """Apply 3-axis RoPE to Q/K tensors.

    Args:
        x: Input tensor [B, N, D] where D = sum(axes_dim).
        cos: Cosine frequencies [B, N, D].
        sin: Sine frequencies [B, N, D].

    Returns:
        RoPE-applied tensor [B, N, D].
    """
    # Split x into two halves (standard RoPE applies to pairs)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # Apply rotation: (x1*cos - x2*sin, x1*sin + x2*cos)
    out = mx.concatenate([x1 * cos[..., : x1.shape[-1]] - x2 * sin[..., : x1.shape[-1]],
                          x1 * cos[..., x1.shape[-1]:] + x2 * sin[..., x1.shape[-1]:]], axis=-1)
    return out


def compute_rope_3d(
    seq_len: int,
    axes_dim: tuple[int, int, int] = (32, 48, 48),
    theta: float = 1000.0,
    device: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array]:
    """Compute 3-axis RoPE cos/sin for image tokens.

    Creates position indices where all positions have frame=0 (target tokens).
    Text tokens use all-zero positions (no positional encoding).

    Args:
        seq_len: Sequence length (number of image tokens = H/patch * W/patch).
        axes_dim: (frame_dim, height_dim, width_dim).
        theta: RoPE base frequency.
        device: MLX dtype for output.

    Returns:
        (cos, sin) tensors of shape [seq_len, sum(axes_dim)].
    """
    h_pos = mx.arange(seq_len, dtype=mx.float32)
    w_pos = mx.arange(0, 0, dtype=mx.float32)  # placeholder, not used for 1D seq

    # For a 2D image grid, we'd compute h and w positions separately.
    # For the 1D sequence case (flattened grid), position index serves as both.
    cos_h, sin_h = _rope_axis(h_pos, axes_dim[1], theta)
    cos_w, sin_w = _rope_axis(h_pos, axes_dim[2], theta)  # use same positions for w

    # Frame axis: all zeros (target tokens)
    frame_pos = mx.zeros((1,), dtype=mx.float32)
    cos_f, sin_f = _rope_axis(frame_pos, axes_dim[0], theta)
    cos_f = cos_f[0]  # [axes_dim[0]]
    sin_f = sin_f[0]

    # Build combined cos/sin per position
    cos_all = mx.concatenate([cos_f[None], cos_h * cos_w], axis=-1)
    sin_all = mx.concatenate([sin_f[None], sin_h * sin_w], axis=-1)

    return cos_all.astype(device), sin_all.astype(device)


class EmbedND(nn.Module):
    """N-dimensional RoPE embedding module.

    Generalizes the standard EmbedND to support arbitrary axis dimensions.
    Used by both FLUX.1 (axes [16, 56, 56]) and Krea2 (axes [32, 48, 48]).

    For each position, computes concatenated cos/sin from all axes.
    """

    def __init__(self, axes_dim: tuple[int, ...], theta: float = 1000.0):
        """Initialize EmbedND.

        Args:
            axes_dim: Tuple of dimensions per axis. Must sum to hidden_dim.
            theta: Base frequency for RoPE.
        """
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def __call__(self, pos: mx.array) -> tuple[mx.array, mx.array]:
        """Compute RoPE embeddings for position indices.

        Args:
            pos: Position tensor of shape [B, N, num_axes] where each
                 element is an index for the corresponding axis.

        Returns:
            (cos, sin) tensors of shape [B, N, sum(axes_dim)].
        """
        cos_parts = []
        sin_parts = []

        for axis_idx, axis_dim in enumerate(self.axes_dim):
            axis_pos = pos[..., axis_idx]  # [B, N]
            cos_a, sin_a = _rope_axis(axis_pos, axis_dim, self.theta)
            cos_parts.append(cos_a)
            sin_parts.append(sin_a)

        cos = mx.concatenate(cos_parts, axis=-1)
        sin = mx.concatenate(sin_parts, axis=-1)
        return cos, sin
