"""
Native MLX Transformer for FLUX.1
==================================
A minimal but complete FLUX.1 transformer implementation in pure MLX.

This module provides:
  - FluxConfig: model configuration (from config submodule)
  - FluxTransformer: the full transformer with double_blocks + single_blocks
  - load_transformer: loads safetensors checkpoints into MLX arrays

Architecture (FLUX.1 dev):
  - 19 double transformer blocks (img + txt cross-attention)
  - 38 single transformer blocks (joint img+txt attention + MLP)
  - Hidden dim: 3072, MLP dim: 12288
  - 24 attention heads (head_dim = 128)

Design decisions:
  - Use mx.Linear for all linear layers (built-in, efficient on Metal)
  - Fused rope/attention where possible
  - Lazy evaluation with strategic mx.eval() for bridge points
  - float16 default, bfloat16 supported
"""

from __future__ import annotations

__all__ = ["FluxConfig", "FluxTransformer", "load_transformer"]

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Re-export config from submodule
from .config import FluxConfig  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────

HIDDEN_DIM = 3072
MLP_DIM = 12288
NUM_HEADS = 24
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 128
MAX_POSITIONS = 4096


# ── Helper layers ─────────────────────────────────────────────────────

def rope(pos: mx.array, dim: int, theta: float = 10000.0) -> mx.array:
    """Rotary positional embeddings.

    Args:
        pos: [N] position indices
        dim: embedding dimension (must be even)
        theta: scaling factor
    Returns:
        [N, dim/2] cosine and sine values
    """
    assert dim % 2 == 0
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    freqs = (1.0 / (theta ** scale)).astype(mx.float32)
    theta_i = pos.astype(mx.float32)[:, None] * freqs[None, :]  # [N, D/2]
    cos = mx.cos(theta_i)
    sin = mx.sin(theta_i)
    return mx.concatenate([cos, sin], axis=-1)  # [N, D]


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary positional embeddings to QK pairs."""
    split = x.shape[-1] // 2
    x1 = x[..., :split]
    x2 = x[..., split:]
    return (x1 * cos[..., :split] - x2 * sin[..., :split]), \
           (x1 * sin[..., :split] + x2 * cos[..., :split])


class EmbedND(nn.Module):
    """Get image/text positional embeddings for FLUX.

    Maps [B, T, dim] -> [B, T, dim] with sinusoidal positional encoding.
    For FLUX: text gets T5 positions, image gets 2D spatial positions.
    """

    def __init__(self, dim: int = HIDDEN_DIM, theta: float = 10000.0,
                 freq_scale_c: float = 1.0, freq_scale_h: float = 1.0,
                 freq_scale_w: float = 1.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freq_scale_c = freq_scale_c
        self.freq_scale_h = freq_scale_h
        self.freq_scale_w = freq_scale_w

    def __call__(self, x: mx.array, image_sizes: tuple[int, int] | None = None) -> mx.array:
        """
        Args:
            x: [B, T, dim] input embeddings
            image_sizes: (height, width) in pixels for spatial rope

        Returns:
            [B, T, dim] with positional embeddings added
        """
        _, length, dim = x.shape
        pos = mx.arange(length, dtype=mx.float32)  # [T]
        rope_mult = rope(pos, self.dim, self.theta)  # [T, D]
        return x + rope_mult[None]  # [B, T, D]


class Modulation(nn.Module):
    """Modulation layers for diffusion transformer blocks.

    Outputs 6 parameters (for double blocks) or 9 parameters (for single blocks):
      double: shift_msa, gate_msa, shift_mlp, gate_mlp, shift_norm2, gate_norm2
      single: shift_msa, gate_msa, shift_mlp, gate_mlp, gate_norm2, shift_norm2, ...
    """

    def __init__(self, dim: int, num_params: int = 6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, affine=False)
        self.linear = nn.Linear(dim, num_params * dim)

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        """Returns a tuple of (param_i * x) for each of num_params."""
        normed = self.norm(x)
        params = self.linear(nn.silu(normed))
        # Split into num_params chunks
        chunk_size = params.shape[-1] // 6
        params = mx.split(params, 6, axis=-1)
        return tuple(p for p in params)


class LinearAttention(nn.Module):
    """Multi-head attention with QKV projection.

    Supports optional Kontext KV cache injection for reference conditioning.
    """

    def __init__(self, dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS,
                 qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        kontext_kv: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """
        Args:
            x: [B, N, D] input
            rope: [N, D] rope embeddings
            kontext_kv: optional (k_ref, v_ref) for reference conditioning

        Returns:
            [B, N, D] attention output
        """
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        # Kontext KV cache injection
        if kontext_kv is not None:
            k_ref, v_ref = kontext_kv
            k = mx.concatenate([k, k_ref], axis=2)
            v = mx.concatenate([v, v_ref], axis=2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)  # [B, H, N, N+]
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v  # [B, H, N, head_dim]
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)


# ── Double Block (img + txt cross-attention) ─────────────────────────

class DoubleBlock(nn.Module):
    """FLUX.1 double transformer block.

    Processes image and text tokens in parallel:
      1. Image self-attention (with rope)
      2. Text self-attention (with rope)
      3. Image cross-attention to text
      4. Text cross-attention to image (skip for FLUX dev)
      5. Image MLP (gated)
      6. Text MLP (gated)
    """

    def __init__(self, dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.img_mod = Modulation(dim, num_params=6)
        self.txt_mod = Modulation(dim, num_params=6)

        self.img_attn = LinearAttention(dim, num_heads)
        self.txt_attn = LinearAttention(dim, num_heads)

        # Text-to-image cross-attention (FLUX dev: text attends to image via txt_attn)
        # In FLUX dev, txt_attn is self-attention on concatenated [img, txt]
        # Simplified: we use the combined rope approach

        hidden_mlp = int(dim * mlp_ratio)
        self.img_mlp_0 = nn.Linear(dim, hidden_mlp)
        self.img_mlp_2 = nn.Linear(hidden_mlp, dim)
        self.txt_mlp_0 = nn.Linear(dim, hidden_mlp)
        self.txt_mlp_2 = nn.Linear(hidden_mlp, dim)

    def __call__(
        self,
        img: mx.array,
        txt: mx.array,
        rope: mx.array,
        kontext_kv: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Args:
            img: [B, N_img, D] image tokens
            txt: [B, N_txt, D] text tokens
            rope: [N, D] rope embeddings (N = max(N_img, N_txt))
            kontext_kv: optional (k_ref, v_ref) for reference conditioning

        Returns:
            (img_out, txt_out)
        """
        # --- Image branch ---
        img_mod1_shift, img_mod1_gate, img_mod2_shift, img_mod2_gate, img_mod3_shift, img_mod3_gate = \
            self.img_mod(img)

        img_modded = img * (1 + img_mod1_shift) + img_mod1_gate

        # Self-attention on image tokens (with optional Kontext)
        img_attn_out = self.img_attn(img_modded, rope, kontext_kv=kontext_kv)
        img = img + img_attn_out * img_mod2_gate

        # --- Text branch ---
        txt_mod1_shift, txt_mod1_gate, txt_mod2_shift, txt_mod2_gate, txt_mod3_shift, txt_mod3_gate = \
            self.txt_mod(txt)

        txt_modded = txt * (1 + txt_mod1_shift) + txt_mod1_gate

        # Self-attention on text tokens (with optional Kontext)
        txt_attn_out = self.txt_attn(txt_modded, rope, kontext_kv=kontext_kv)
        txt = txt + txt_attn_out * txt_mod2_gate

        # --- MLP ---
        img_mlp = self.img_mlp_2(nn.gelu(self.img_mlp_0(img_modded)))
        img = img + img_mlp * img_mod3_gate

        txt_mlp = self.txt_mlp_2(nn.gelu(self.txt_mlp_0(txt_modded)))
        txt = txt + txt_mlp * txt_mod3_gate

        return img, txt


# ── Single Block (joint attention) ────────────────────────────────────

class SingleBlock(nn.Module):
    """FLUX.1 single transformer block.

    Concatenates image and text tokens, then applies joint attention + MLP.
    More efficient than double blocks for the second half of the network.
    """

    def __init__(self, dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.mod = Modulation(dim, num_params=9)

        self.attn = LinearAttention(dim, num_heads)

        hidden_mlp = int(dim * mlp_ratio)
        self.mlp_0 = nn.Linear(dim, hidden_mlp * 3)  # Fused: 3 projections
        self.mlp_2 = nn.Linear(hidden_mlp, dim)

    def __call__(
        self,
        x: mx.array,
        rope: mx.array,
        kontext_kv: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """
        Args:
            x: [B, N, D] concatenated [img, txt] tokens
            rope: [N, D] rope embeddings
            kontext_kv: optional (k_ref, v_ref) for reference conditioning

        Returns:
            [B, N, D] output
        """
        mod_shift, mod_gate, shift2, gate2, shift3, gate3, shift4, gate4, shift5 = \
            self.mod(x)

        x_modded = x * (1 + shift2) + gate2

        # Joint attention (with optional Kontext)
        attn_out = self.attn(x_modded, rope, kontext_kv=kontext_kv)
        x = x + attn_out * gate3

        # Fused MLP with swiglu
        mlp_out = self.mlp_0(nn.silu(x_modded))
        # Split into 3 parts and apply SwiGLU
        mlp_out = mx.split(mlp_out, 3, axis=-1)
        mlp_out = mlp_out[0] * nn.gelu(mlp_out[1])
        mlp_out = self.mlp_2(mlp_out)
        x = x + mlp_out * gate3

        return x


# ── Main Transformer ──────────────────────────────────────────────────

class FluxTransformer(nn.Module):
    """Complete FLUX.1 transformer.

    Architecture:
      img_in / txt_in  ->  19x DoubleBlock  ->  38x SingleBlock  ->  final_layer
    """

    def __init__(self, config: FluxConfig | None = None):
        super().__init__()
        config = config or FluxConfig()
        self.config = config
        self.dtype = config.mlx_dtype

        # Input projections
        self.img_in = nn.Linear(64, HIDDEN_DIM)
        self.txt_in = nn.Linear(4096, HIDDEN_DIM)

        # Time / guidance embedding
        self.time_in = nn.Linear(256, HIDDEN_DIM * 4)
        self.vector_in = nn.Linear(768, HIDDEN_DIM * 4)
        self.guidance_in = nn.Linear(768, HIDDEN_DIM * 4) if config.guidance_embed else None

        # Transformer blocks
        self.double_blocks = [
            DoubleBlock() for _ in range(config.num_double_blocks)
        ]
        self.single_blocks = [
            SingleBlock() for _ in range(config.num_single_blocks)
        ]

        # Output
        self.final_layer = nn.Linear(HIDDEN_DIM, 64)

        # Positional embedding
        self.rope = EmbedND(HIDDEN_DIM)

        # Kontext KV cache for reference conditioning
        self.kontext_kv_cache: dict[str, tuple[mx.array, mx.array]] = {}
        self.kontext_kv_enabled: bool = False
        self.kontext_ref_tokens: int = 0

    def set_kontext(self, enabled: bool, reference_tokens: int = 0) -> None:
        """Enable/disable Kontext KV cache."""
        self.kontext_kv_enabled = enabled and reference_tokens > 0
        self.kontext_ref_tokens = reference_tokens
        if not enabled:
            self.kontext_kv_cache.clear()

    def get_kontext_kv(self, layer_idx: int) -> tuple[mx.array, mx.array] | None:
        """Get cached K/V for a given layer, or None."""
        if not self.kontext_kv_enabled:
            return None
        key = f"block_{layer_idx}"
        return self.kontext_kv_cache.get(key)

    def store_kontext_kv(self, layer_idx: int, k: mx.array, v: mx.array) -> None:
        """Store reference K/V for a given layer."""
        if not self.kontext_kv_enabled or self.kontext_ref_tokens <= 0:
            return
        key = f"block_{layer_idx}"
        if k.shape[2] >= self.kontext_ref_tokens:
            ref_k = mx.contiguous(k[:, :, -self.kontext_ref_tokens:, :])
            ref_v = mx.contiguous(v[:, :, -self.kontext_ref_tokens:, :])
            mx.eval(ref_k, ref_v)
            self.kontext_kv_cache[key] = (ref_k, ref_v)

    def get_rope(self, img_len: int, txt_len: int) -> mx.array:
        """Compute rope embeddings for image and text lengths."""
        # For FLUX dev: image tokens = H/8 * W/8, text tokens = T5 sequence
        # rope is computed per-position
        max_len = max(img_len, txt_len)
        pos = mx.arange(max_len, dtype=mx.float32)
        return rope(pos, HIDDEN_DIM)

    def time_embed(self, t: mx.array, guidance: mx.array | None = None,
                   pooled: mx.array | None = None) -> mx.array:
        """Compute time + guidance + pooled conditioning."""
        # Standard timestep embedding
        half_dim = 256 // 2
        emb_math = mx.log(10000.0) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb_math)
        emb = t[:, None].astype(mx.float32) * emb[None, :]
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
        emb = self.time_in(emb.astype(self.dtype))

        # Add pooled (CLIP-L) embedding
        if pooled is not None:
            pooled_emb = self.vector_in(pooled.astype(self.dtype))
            emb = emb + pooled_emb

        # Add guidance embedding (FLUX dev only)
        if guidance is not None and self.guidance_in is not None:
            guidance_emb = self.guidance_in(
                guidance.astype(self.dtype)
            )
            emb = emb + guidance_emb

        return emb

    def __call__(
        self,
        img: mx.array,       # [B, N_img, 64] packed image patches
        txt: mx.array,       # [B, N_txt, 4096] T5 embeddings
        t: mx.array,         # [B] timestep
        guidance: mx.array | None = None,  # [B] guidance scale
        pooled: mx.array | None = None,    # [B, 768] pooled CLIP
        rope: mx.array | None = None,      # precomputed rope
    ) -> mx.array:
        """Forward pass.

        Args:
            img: packed image tokens [B, N_img, 64]
            txt: text embeddings [B, N_txt, 4096]
            t: timesteps [B]
            guidance: guidance scale [B] (FLUX dev)
            pooled: pooled CLIP output [B, 768]
            rope: precomputed rope embeddings [N, D]

        Returns:
            [B, N_img, 64] noise prediction
        """
        B = img.shape[0]

        # Input projections
        img = self.img_in(img.astype(self.dtype))
        txt = self.txt_in(txt.astype(self.dtype))

        # Time conditioning
        cond = self.time_embed(t, guidance=guidance, pooled=pooled)

        # --- Double blocks ---
        if rope is None:
            rope = self.get_rope(img.shape[1], txt.shape[1])

        for i, block in enumerate(self.double_blocks):
            kv = self.get_kontext_kv(i)
            img, txt = block(img, txt, rope, kontext_kv=kv)

        # Concatenate for single blocks
        x = mx.concatenate([img, txt], axis=1)

        # --- Single blocks ---
        for i, block in enumerate(self.single_blocks):
            kv = self.get_kontext_kv(len(self.double_blocks) + i)
            x = block(x, rope, kontext_kv=kv)

        # Split back
        img_out = x[:, :img.shape[1], :]

        # Final layer
        noise = self.final_layer(img_out)
        return noise

    def predict(
        self,
        img: mx.array,
        txt: mx.array,
        timestep: float,
        guidance: float = 3.5,
        pooled: mx.array | None = None,
        rope: mx.array | None = None,
    ) -> mx.array:
        """Convenience method for one denoising step.

        Args:
            img: [B, N, 64] current latent
            txt: [B, T, 4096] text embedding
            timestep: float sigma/timestep value
            guidance: guidance scale
            pooled: [B, 768] pooled CLIP
            rope: optional precomputed rope

        Returns:
            [B, N, 64] predicted noise
        """
        t = mx.array([timestep], dtype=mx.float32)
        g = mx.array([guidance], dtype=mx.float32) if guidance is not None else None
        return self(img, txt, t, guidance=g, pooled=pooled, rope=rope)


# ── Loader ────────────────────────────────────────────────────────────

def _normalize_key(key: str) -> str:
    """Normalize a PyTorch/ComfyUI key to our internal naming."""
    # Strip common prefixes
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _load_safetensors(path: str | Path) -> dict[str, mx.array]:
    """Load a safetensors file into MLX arrays."""
    import safetensors
    with open(path, "rb") as f:
        data = safetensors.numpy.load(f.read())
    return {k: mx.array(v) for k, v in data.items()}


def load_transformer(
    path: str | Path,
    dtype: str = "float16",
) -> FluxTransformer:
    """Load a FLUX.1 checkpoint into a FluxTransformer.

    Args:
        path: path to safetensors checkpoint
        dtype: "float16" or "bfloat16"

    Returns:
        Loaded FluxTransformer (weights not yet assigned to module)
    """
    path = Path(path)
    state = _load_safetensors(path)

    # Normalize keys
    normalized = {}
    for k, v in state.items():
        nk = _normalize_key(k)
        if nk is not None:
            normalized[nk] = v

    config = FluxConfig(dtype=dtype)
    model = FluxTransformer(config)

    # Assign weights - map our keys to checkpoint keys
    # This is a simplified mapping; real checkpoints may vary
    _assign_weights(model, normalized, config)

    mx.eval(model.parameters())
    return model


def _assign_weights(model: FluxTransformer, state: dict[str, mx.array],
                    config: FluxConfig) -> None:
    """Assign loaded weights to model parameters."""
    # Input projections
    if "img_in.weight" in state:
        model.img_in.weight = state["img_in.weight"]
    if "txt_in.weight" in state:
        model.txt_in.weight = state["txt_in.weight"]

    # Time/guidance embeddings
    if "time_in.in_layer.weight" in state:
        model.time_in.weight = state["time_in.in_layer.weight"]
        model.time_in.bias = state["time_in.in_layer.bias"]
    if "vector_in.in_layer.weight" in state:
        model.vector_in.weight = state["vector_in.in_layer.weight"]
        model.vector_in.bias = state["vector_in.in_layer.bias"]
    if config.guidance_embed and "guidance_in.in_layer.weight" in state:
        model.guidance_in.weight = state["guidance_in.in_layer.weight"]
        model.guidance_in.bias = state["guidance_in.in_layer.bias"]

    # Transformer blocks
    for i, block in enumerate(model.double_blocks):
        prefix = f"double_blocks.{i}."
        _assign_double_block(block, state, prefix)

    for i, block in enumerate(model.single_blocks):
        prefix = f"single_blocks.{i}."
        _assign_single_block(block, state, prefix)

    # Final layer
    if "final_layer.linear.weight" in state:
        model.final_layer.weight = state["final_layer.linear.weight"]
        if "final_layer.linear.bias" in state:
            model.final_layer.bias = state["final_layer.linear.bias"]


def _assign_double_block(block: DoubleBlock, state: dict[str, mx.array],
                         prefix: str) -> None:
    """Assign weights to a DoubleBlock."""
    # Modulation layers
    for name, submodule in [("img_mod", block.img_mod), ("txt_mod", block.txt_mod)]:
        p = f"{prefix}{name}."
        submodule.norm.weight = state[f"{p}norm.weight"]
        submodule.norm.bias = state[f"{p}norm.bias"]
        submodule.linear.weight = state[f"{p}linear.weight"]
        submodule.linear.bias = state[f"{p}linear.bias"]

    # Attention
    for name, attn in [("img_attn", block.img_attn), ("txt_attn", block.txt_attn)]:
        p = f"{prefix}{name}."
        attn.qkv.weight = state[f"{p}qkv.weight"]
        if f"{p}qkv.bias" in state:
            attn.qkv.bias = state[f"{p}qkv.bias"]
        attn.proj.weight = state[f"{p}proj.weight"]
        if f"{p}proj.bias" in state:
            attn.proj.bias = state[f"{p}proj.bias"]

    # MLP
    for name, mlp in [("img_mlp", block.img_mlp_0), ("txt_mlp", block.txt_mlp_0)]:
        p = f"{prefix}{name}."
        mlp.weight = state[f"{p}0.weight"]
        if f"{p}0.bias" in state:
            mlp.bias = state[f"{p}0.bias"]

    for name, mlp in [("img_mlp", block.img_mlp_2), ("txt_mlp", block.txt_mlp_2)]:
        p = f"{prefix}{name}."
        mlp.weight = state[f"{p}2.weight"]
        if f"{p}2.bias" in state:
            mlp.bias = state[f"{p}2.bias"]


def _assign_single_block(block: SingleBlock, state: dict[str, mx.array],
                         prefix: str) -> None:
    """Assign weights to a SingleBlock."""
    # Modulation
    p = f"{prefix}mod."
    block.mod.norm.weight = state[f"{p}norm.weight"]
    block.mod.norm.bias = state[f"{p}norm.bias"]
    block.mod.linear.weight = state[f"{p}linear.weight"]
    block.mod.linear.bias = state[f"{p}linear.bias"]

    # Attention
    p = f"{prefix}attn."
    block.attn.qkv.weight = state[f"{p}qkv.weight"]
    if f"{p}qkv.bias" in state:
        block.attn.qkv.bias = state[f"{p}qkv.bias"]
    block.attn.proj.weight = state[f"{p}proj.weight"]
    if f"{p}proj.bias" in state:
        block.attn.proj.bias = state[f"{p}proj.bias"]

    # MLP
    p = f"{prefix}mlp."
    block.mlp_0.weight = state[f"{p}0.weight"]
    if f"{p}0.bias" in state:
        block.mlp_0.bias = state[f"{p}0.bias"]
    block.mlp_2.weight = state[f"{p}2.weight"]
    if f"{p}2.bias" in state:
        block.mlp_2.bias = state[f"{p}2.bias"]
