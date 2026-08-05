"""
Krea2 SingleStreamDiT transformer model.

Implements the core architecture components as defined in the ComfyUI reference:
- RMSNorm: RMS normalization with (1 + scale) weight convention
- QKNorm: Per-head Q/K RMSNorm for attention
- SwiGLU: SwiGLU MLP with separate gate/up/down projections
- Attention: GQA (48 Q + 12 K/V) + per-head QK norm + sigmoid-gated output
- DoubleSharedModulation: timestep vec → 6 modulation params (simple add + chunk)
- SimpleModulation: timestep vec → 2 params (scale, shift)
- TextFusionBlock: RMSNorm + attention + SwiGLU MLP
- TextFusionTransformer: 2 layerwise blocks + Linear(12,1) projector + 2 refiner blocks
- SingleStreamBlock: modulation + prenorm + attention + postnorm + MLP
- LastLayer: RMSNorm + SimpleModulation + linear
- SingleStreamDiT: full transformer with img_in, txtfusion, txtmlp, 28 blocks, last

Architecture reference:
  https://github.com/Comfy-Org/ComfyUI/blob/main/comfy/ldm/krea2/model.py
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ── RMSNorm with (1 + scale) convention ─────────────────────────────────

class RMSNorm(nn.Module):
    """RMSNorm with the (1 + scale) weight convention.

    The stored scale is zero-centered, and the actual weight is (1 + scale).
    This matches the ComfyUI reference implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = mx.zeros(dim)  # stored zero-centered

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x_f32 = x.astype(mx.float32)
        # weight = 1 + scale (scale is stored zero-centered)
        weight = 1.0 + self.scale.astype(mx.float32)
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        variance = (x_f32 * x_f32).mean(axis=-1, keepdims=True)
        normed = x_f32 / (variance + self.eps).sqrt()
        return (normed * weight).astype(dtype)


# ── QK Normalization ────────────────────────────────────────────────────

class QKNorm(nn.Module):
    """Per-head Q/K normalization for attention.

    Applies separate RMSNorm to Q and K tensors after projection.
    Each normalizes over the head_dim dimension (128 for Krea2).
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.qnorm = RMSNorm(head_dim)
        self.knorm = RMSNorm(head_dim)

    def __call__(self, q: mx.array, k: mx.array) -> tuple[mx.array, mx.array]:
        return self.qnorm(q), self.knorm(k)


# ── SwiGLU MLP ──────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate) * up).

    MLP dimension: ceil((2/3 * features) * multiplier / 128) * 128
    For features=6144, multiplier=4: mlpdim=6720
    """

    def __init__(self, features: int, multiplier: int = 4, multiple: int = 128):
        super().__init__()
        mlp_dim = int(2 * features / 3) * multiplier
        mlp_dim = multiple * ((mlp_dim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlp_dim)
        self.up = nn.Linear(features, mlp_dim)
        self.down = nn.Linear(mlp_dim, features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


# ── Attention with GQA + QK Norm + Sigmoid Gate ─────────────────────────

class Attention(nn.Module):
    """GQA attention with per-head QK norm and sigmoid-gated output.

    Architecture:
      wq: Linear(dim, heads * head_dim)     # 48 heads
      wk: Linear(dim, kvheads * head_dim)   # 12 heads
      wv: Linear(dim, kvheads * head_dim)   # 12 heads
      gate: Linear(dim, dim)                # gating vector
      qknorm: QKNorm(head_dim)              # per-head Q/K normalization
      wo: Linear(dim, dim)                  # output projection
    """

    def __init__(self, dim: int, heads: int, kvheads: int | None = None):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads  # 6144 / 48 = 128

        self.wq = nn.Linear(dim, self.headdim * self.heads)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads)
        self.gate_proj = nn.Linear(dim, dim)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        freqs: mx.array | None = None,
        ref_boost: mx.array | None = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Project Q, K, V — reshape to (B, L, H, D) then transpose to (B, H, L, D)
        # NOTE: reshape must first split the last dim (B,L,H,D), then transpose to move H axis
        q = self.wq(x).reshape(B, L, self.heads, self.headdim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.kvheads, self.headdim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.kvheads, self.headdim).transpose(0, 2, 1, 3)

        # Per-head Q/K normalization
        q, k = self.qknorm(q, k)

        # Apply RoPE if provided (paired-interleave convention, matching
        # comfy.ldm.flux.math.apply_rope which the Krea2 reference reuses)
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        # GQA: repeat K/V heads to match Q heads
        # After transpose (B, H, L, D), heads are on axis=1
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.headdim)
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)  # [B, H, L, L]

        # ref_boost: additive attention-logit bias
        if ref_boost is not None:
            attn = attn + ref_boost

        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = attn @ v  # [B, H, L, headdim]

        # Reshape out to [B, L, D] for elementwise gate multiplication
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)  # [B, L, D]

        # Sigmoid-gated output — gate has the same shape as out
        gate = mx.sigmoid(self.gate_proj(x))  # [B, L, D]
        out = out * gate  # [B, L, D]

        return self.wo(out)


# ── Modulation Layers ───────────────────────────────────────────────────

class DoubleSharedModulation(nn.Module):
    """Timestep modulation: vec + lin → 6 params (chunk).

    Simple additive modulation: out = vec + lin, then chunk into 6.
    This is the Krea2 convention, NOT the ComfyUI-MLXU convention.

    Parameters:
      lin: [6 * dim] — additive modulation parameters
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((6 * dim,))  # stored as parameter

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Compute 6 modulation parameters from timestep embedding.

        Args:
            vec: Timestep embedding [B, 6*dim].

        Returns:
            Tuple of (prescale, preshift, pregate, postscale, postshift, postgate),
            each of shape [B, dim].
        """
        out = vec + self.lin.astype(vec.dtype)[None, :]
        B = out.shape[0]
        chunks = out.split(6, axis=-1)
        return tuple(c.reshape(B, -1) for c in chunks)


class SimpleModulation(nn.Module):
    """Timestep modulation: vec + lin → 2 params (scale, shift).

    Used in the LastLayer for final output modulation. Matches the ComfyUI
    reference: `lin` is a [2, dim] parameter added to `vec` via broadcast
    (vec[B,dim] + lin[2,dim] -> [B,2,dim] out), then split along the new
    axis into scale/shift.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lin = mx.zeros((2, dim))

    def __call__(self, vec: mx.array) -> tuple[mx.array, mx.array]:
        """Compute scale and shift from timestep embedding.

        Args:
            vec: Timestep embedding [B, dim].

        Returns:
            Tuple of (scale, shift), each of shape [B, dim].
        """
        out = vec[:, None, :] + self.lin.astype(vec.dtype)[None, :, :]  # [B, 2, dim]
        scale, shift = out[:, 0, :], out[:, 1, :]
        return scale, shift


# ── TextFusionBlock ─────────────────────────────────────────────────────

class TextFusionBlock(nn.Module):
    """Text fusion block: RMSNorm + attention + SwiGLU MLP.

    Uses separate pre and post normalization (no timestep modulation).
    """

    def __init__(self, features: int, heads: int, kvheads: int | None = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads)
        # Text fusion uses multiplier=4 for SwiGLU (same as main blocks)
        self.mlp = SwiGLU(features, multiplier=4)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.prenorm(x))
        x = x + self.mlp(self.postnorm(x))
        return x


# ── TextFusionTransformer ──────────────────────────────────────────────

class TextFusionTransformer(nn.Module):
    """Text fusion adapter for Qwen3-VL layer taps.

    Architecture:
      Input:  [B, seq, txtlayers, txtdim]  (unpacked from stacked layers)
      2 layerwise blocks (process per-layer)
      Rearrange: [B, seq, txtlayers, txtdim] → [B, seq, txtdim, txtlayers]
      Projector: Linear(txtlayers, 1) → reduces layer dim
      2 refiner blocks (process combined text)
      Output: [B, seq, txtdim]

    The projector Linear(12, 1) reduces the 12-layer stacked representation
    to a single-layer representation by fusing across the layer dimension.
    """

    def __init__(self, num_txt_layers: int = 12, text_dim: int = 2560,
                 heads: int = 20, kvheads: int | None = None):
        super().__init__()
        self.num_txt_layers = num_txt_layers
        self.text_dim = text_dim
        self.layerwise_blocks = [
            TextFusionBlock(text_dim, heads, kvheads) for _ in range(2)
        ]
        # Projector: Linear(num_txt_layers, 1) — fuses layer dimension
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = [
            TextFusionBlock(text_dim, heads, kvheads) for _ in range(2)
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


# ── SingleStreamBlock ───────────────────────────────────────────────────

class SingleStreamBlock(nn.Module):
    """Krea2 single-stream transformer block.

    Architecture:
      Modulation: DoubleSharedModulation(timestep_vec) → 6 params
      Pre-norm: RMSNorm → modulated by (1+prescale) and preshift
      Attention: GQA + QK norm + RoPE + sigmoid gate
      Post-norm: RMSNorm → modulated by (1+postscale) and postshift
      MLP: SwiGLU → gated by postgate

      x + pregate * attn((1+prescale) * prenorm(x) + preshift)
      x + postgate * mlp((1+postscale) * postnorm(x) + postshift)

    For Identity Edit, the SAME timestep vector modulates the entire sequence
    (text, source, and target tokens alike) — only the RoPE frame index
    (source=1, target=0) distinguishes source from target. There is no
    per-token modulation split; matching the ComfyUI reference forward
    (krea2_edit_forward), a single `tvec` is used throughout.
    """

    def __init__(self, features: int, heads: int, multiplier: int = 4, kvheads: int | None = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads)
        self.mlp = SwiGLU(features, multiplier)

    def __call__(
        self,
        x: mx.array,
        vec: mx.array,
        freqs: mx.array | None = None,
        ref_boost: mx.array | None = None,
    ) -> mx.array:
        """Forward pass through single-stream block.

        Args:
            x: Input tensor [B, N, features].
            vec: Timestep embedding [B, features], shared by all tokens in the sequence.
            freqs: [N, head_dim/2, 2, 2] RoPE rotation table.
            ref_boost: Attention logit bias for reference conditioning.

        Returns:
            Output tensor [B, N, features].
        """
        # Get 6 modulation parameters from timestep vector
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)

        # Modulation params are [B, D] -> [B, 1, D] for broadcasting with [B, N, D]
        x = x + pregate[:, None] * self.attn(
            (1 + prescale[:, None]) * self.prenorm(x) + preshift[:, None],
            freqs, ref_boost,
        )
        x = x + postgate[:, None] * self.mlp(
            (1 + postscale[:, None]) * self.postnorm(x) + postshift[:, None],
        )

        return x


# ── LastLayer ───────────────────────────────────────────────────────────

class LastLayer(nn.Module):
    """Final layer: RMSNorm + SimpleModulation + linear.

    Applies timestep-based scale/shift to normalized features,
    then projects to output dimension.
    """

    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.modulation = SimpleModulation(features)
        self.linear = nn.Linear(features, patch * patch * channels)

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        """Forward pass through last layer.

        Args:
            x: Input tensor [B, N, features].
            vec: Timestep embedding [B, features].

        Returns:
            Output tensor [B, N, patch*patch*channels].
        """
        scale, shift = self.modulation(vec)
        # scale/shift are [B, D] -> [B, 1, D] for broadcasting with [B, N, D]
        x = (1 + scale[:, None]) * self.norm(x) + shift[:, None]
        return self.linear(x)


# ── SingleStreamDiT (Krea2Transformer) ──────────────────────────────────

class SingleStreamDiT(nn.Module):
    """Complete Krea2 SingleStreamDiT transformer.

    Architecture:
      img_in (latent_channels*patch² → features)     Image projection
      txtfusion (txtlayers*txtdim → txtdim)           Text fusion adapter
      txtmlp (txtdim → features)                       Text MLP
      tmlp (tdim → features → features)                Time MLP
      tproj (features → features*6)                     Time projection
      28x SingleStreamBlock                             Transformer blocks
      last (features → patch²*channels)                 Output layer
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = config.mlx_dtype
        self.patch = config.patch_size
        self.channels = config.latent_channels
        self.tdim = config.time_dim
        self.heads = config.num_heads
        self.txtdim = config.text_dim
        self.txtlayers = config.text_layers
        self.num_blocks = config.num_blocks

        headdim = config.head_dim  # 128
        self.rope_axes_dim = config.rope_axes_dim  # (128, 128, 128)
        self.rope_theta = config.rope_theta

        # Input projections
        self.first = nn.Linear(config.latent_channels * config.patch_size ** 2, config.hidden_dim)

        # Transformer blocks
        self.blocks = [
            SingleStreamBlock(
                config.hidden_dim,
                config.num_heads,
                multiplier=4,
                kvheads=config.num_kv_heads,
            )
            for _ in range(config.num_blocks)
        ]

        # Time MLP: tdim → features → features (GELU)
        self.tmlp = nn.Sequential(
            nn.Linear(config.time_dim, config.hidden_dim),
            nn.GELU(approx="tanh"),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Text fusion transformer
        self.txtfusion = TextFusionTransformer(
            num_txt_layers=config.text_layers,
            text_dim=config.text_dim,
            heads=20,
            kvheads=20,
        )

        # Text MLP: txtdim → features → features (RMSNorm + GELU)
        self.txtmlp = nn.Sequential(
            RMSNorm(config.text_dim),
            nn.Linear(config.text_dim, config.hidden_dim),
            nn.GELU(approx="tanh"),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Output layer
        self.last = LastLayer(config.hidden_dim, config.patch_size, config.latent_channels)

        # Time projection: features → features*6 (GELU, not SiLU)
        self.tproj = nn.Sequential(
            nn.GELU(approx="tanh"),
            nn.Linear(config.hidden_dim, config.hidden_dim * 6),
        )

        # RoPE embedding module
        self.pos_embedder = EmbedND(config.rope_axes_dim, config.rope_theta)

    def time_embed(self, t: mx.array) -> tuple[mx.array, mx.array]:
        """Compute timestep embedding and its 6-param modulation projection.

        Matches the ComfyUI reference: `t = tmlp(timestep_embedding(...))` is
        used directly by LastLayer's SimpleModulation, while `tvec = tproj(t)`
        (6x wider) drives DoubleSharedModulation in every transformer block.

        Args:
            t: Timestep tensor [B].

        Returns:
            (t_embed, tvec): t_embed is [B, hidden_dim], tvec is [B, 6*hidden_dim].
        """
        # Standard sinusoidal timestep embedding (time_factor=1000, matching
        # comfy.ldm.flux.layers.timestep_embedding used by the Krea2 reference)
        half_dim = self.config.time_dim // 2
        emb_math = mx.log(mx.array(10000.0, dtype=mx.float32)) / (half_dim - 1)
        freqs = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb_math)
        emb = (1000.0 * t)[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)  # [B, 256]

        t_embed = self.tmlp(emb.astype(self.dtype))  # [B, hidden]
        tvec = self.tproj(t_embed)  # [B, 6*hidden]
        return t_embed, tvec

    def get_rope(
        self,
        seq_len: int,
        txt_len: int = 0,
        src_len: int = 0,
    ) -> tuple[mx.array, mx.array]:
        raise NotImplementedError(
            "get_rope(seq_len, txt_len, src_len) is deprecated: it cannot express a "
            "2D image grid from a flat token count. Use get_rope_grid(img_h, img_w, "
            "txt_len, src_grids) instead."
        )

    def get_rope_grid(
        self,
        img_h: int,
        img_w: int,
        txt_len: int = 0,
        src_grids: list[tuple[int, int]] | None = None,
    ) -> mx.array:
        """Precompute the RoPE rotation table for a proper 2D image grid.

        Matches the ComfyUI reference (`_imgids` / `krea2_edit_forward`): each
        image token gets its OWN (height, width) grid coordinate, not a flat
        sequential index reused on both axes. Text tokens sit at position 0 on
        every axis; source (Identity Edit) tokens use frame=1 with their own
        grid; target tokens use frame=0.

        Args:
            img_h: Target image token grid height (in patches).
            img_w: Target image token grid width (in patches).
            txt_len: Number of text tokens.
            src_grids: List of (h, w) grid sizes for each source/reference block,
                       in frame order (frame=1, 2, ...). Empty/None for no Identity Edit.

        Returns:
            [txt_len + sum(src_h*src_w) + img_h*img_w, head_dim/2, 2, 2] rotation table.
        """
        src_grids = src_grids or []

        pos_parts = [mx.zeros((txt_len, 3), dtype=mx.float32)] if txt_len > 0 else []

        for frame_idx, (sh, sw) in enumerate(src_grids, start=1):
            pos_parts.append(self._grid_positions(frame_idx, sh, sw))

        pos_parts.append(self._grid_positions(0, img_h, img_w))

        pos = mx.concatenate(pos_parts, axis=0) if len(pos_parts) > 1 else pos_parts[0]
        return self.pos_embedder(pos)

    @staticmethod
    def _grid_positions(frame: int, h: int, w: int) -> mx.array:
        """Build [h*w, 3] position indices (frame, row, col) for a 2D grid."""
        rows = mx.arange(h, dtype=mx.float32)[:, None]
        cols = mx.arange(w, dtype=mx.float32)[None, :]
        rows = mx.broadcast_to(rows, (h, w)).reshape(-1)
        cols = mx.broadcast_to(cols, (h, w)).reshape(-1)
        frames = mx.full((h * w,), float(frame), dtype=mx.float32)
        return mx.stack([frames, rows, cols], axis=-1)

    def unpack_context(self, context: mx.array) -> mx.array:
        """Unpack context from [B, seq, txtlayers*txtdim] to [B, seq, txtlayers, txtdim].

        Args:
            context: Fused text embeddings [B, seq, txtlayers*txtdim].

        Returns:
            Unpacked context [B, seq, txtlayers, txtdim].
        """
        B, seq, fused = context.shape
        expected = self.txtlayers * self.txtdim
        if fused != expected:
            raise ValueError(
                f"Krea2 expects context with {expected} features "
                f"({self.txtlayers}x{self.txtdim}) but got {fused}."
            )
        return context.reshape(B, seq, self.txtlayers, self.txtdim)

    def __call__(
        self,
        img: mx.array,       # [B, N_img, latent_channels*patch²] packed image patches
        txt: mx.array,       # [B, seq, txtlayers*txtdim] fused text embeddings
        t: mx.array,         # [B] timestep
        img_h: int,
        img_w: int,
        freqs: mx.array | None = None,
        ref_boost: mx.array | None = None,
        src: mx.array | None = None,      # [B, N_src, latent_channels*patch²] source latent
        src_h: int | None = None,
        src_w: int | None = None,
    ) -> mx.array:
        """Forward pass through Krea2 transformer.

        Sequence layout matches the ComfyUI reference (`krea2_edit_forward`):
        [text | source(frame=1) | target(frame=0)]. A single timestep vector
        modulates the whole sequence — source/target are distinguished purely
        by the RoPE frame index, not by separate modulation parameters.

        Args:
            img: Packed image latent [B, N_img, 64] (N_img = img_h * img_w).
            txt: Fused text embeddings [B, seq, txtlayers*txtdim].
            t: Timestep [B].
            img_h: Target image token grid height (in patches).
            img_w: Target image token grid width (in patches).
            freqs: Precomputed RoPE rotation table. Computed from img_h/img_w
                   /src_h/src_w if not given.
            ref_boost: Attention logit bias.
            src: Source latent for Identity Edit [B, N_src, 64] (N_src = src_h * src_w).
            src_h: Source token grid height (in patches). Required if src is given.
            src_w: Source token grid width (in patches). Required if src is given.

        Returns:
            Output tensor [B, N_img, latent_channels*patch²].
        """
        # Image projection
        img = self.first(img.astype(self.dtype))
        if img.shape[1] != img_h * img_w:
            raise ValueError(
                f"img_h*img_w ({img_h}*{img_w}={img_h * img_w}) does not match "
                f"img token count ({img.shape[1]})"
            )

        # Text processing: unpack → txtfusion → txtmlp
        context = self.unpack_context(txt)
        context = self.txtfusion(context)
        context = self.txtmlp(context)
        txt_len = context.shape[1]

        # Source latent prepending for Identity Edit
        if src is not None:
            if src_h is None or src_w is None:
                raise ValueError("src_h and src_w are required when src is given")
            src = self.first(src.astype(self.dtype))
            src_len = src.shape[1]
            combined = mx.concatenate([context, src, img], axis=1)
        else:
            combined = mx.concatenate([context, img], axis=1)
            src_len = 0

        # Time embedding — shared by the whole sequence (text, source, target).
        # t_embed [B, hidden] drives LastLayer's SimpleModulation; tvec [B, 6*hidden]
        # drives DoubleSharedModulation in every transformer block.
        t_embed, tvec = self.time_embed(t)

        # Position embeddings: text at 0, source at frame=1 grid, target at frame=0 grid
        if freqs is None:
            src_grids = [(src_h, src_w)] if src is not None else None
            freqs = self.get_rope_grid(img_h, img_w, txt_len, src_grids)

        # Forward through transformer blocks
        for block in self.blocks:
            combined = block(
                combined,
                tvec,
                freqs,
                ref_boost=ref_boost,
            )

        # Extract target image tokens only (drop text and source tokens)
        final_start = txt_len + src_len
        img_out = combined[:, final_start:, :]

        # Output projection (LastLayer uses t_embed, NOT tvec)
        return self.last(img_out, t_embed)

    def predict(
        self,
        img: mx.array,
        txt: mx.array,
        timestep: float | mx.array,
        img_h: int,
        img_w: int,
        src: mx.array | None = None,
        src_h: int | None = None,
        src_w: int | None = None,
        **kwargs,
    ) -> mx.array:
        """Convenience wrapper for the denoising loop.

        Args:
            img: Packed image latent [B, N_img, 64].
            txt: Fused text embeddings [B, seq, txtlayers*txtdim].
            timestep: Timestep value (float or [B] array).
            img_h: Target image token grid height (in patches).
            img_w: Target image token grid width (in patches).
            src: Source latent for Identity Edit [B, N_src, 64].
            src_h: Source token grid height (in patches). Required if src is given.
            src_w: Source token grid width (in patches). Required if src is given.
            **kwargs: Additional kwargs passed to __call__ (rope, ref_boost, etc.)

        Returns:
            Output tensor [B, N_img, 64].
        """
        if isinstance(timestep, float):
            t = mx.array([timestep], dtype=mx.float32)
        else:
            t = timestep.astype(mx.float32)
        return self(img, txt, t, img_h, img_w, src=src, src_h=src_h, src_w=src_w, **kwargs)


# ── EmbedND (3-axis RoPE, paired-interleave convention) ─────────────────
#
# Krea2's reference forward (comfy/ldm/krea2/model.py) imports EmbedND and
# apply_rope straight from comfy.ldm.flux.{layers,math} — it is FLUX's own
# paired-interleave rotation, not the rotate-half/LLaMA convention. These
# free functions mirror native/__init__.py's rope_freqs/embed_nd/apply_rope
# exactly (duplicated rather than imported, since native/__init__.py itself
# imports this module and a cross-import would be circular).

def _rope_freqs_axis(pos: mx.array, dim: int, theta: float) -> mx.array:
    """[N, dim/2, 2, 2] rotation matrices for one RoPE axis."""
    assert dim % 2 == 0
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    omega = 1.0 / (theta ** scale)
    out = pos.astype(mx.float32)[:, None] * omega[None, :]
    cos, sin = mx.cos(out), mx.sin(out)
    return mx.stack([cos, -sin, sin, cos], axis=-1).reshape(*out.shape, 2, 2)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Apply paired-interleave RoPE rotation to Q or K. x: [B,H,N,D], freqs: [N,D/2,2,2]."""
    B, H, N, D = x.shape
    x_pairs = x.reshape(B, H, N, D // 2, 1, 2)
    f = freqs[None, None]
    out = (f[..., 0] * x_pairs[..., 0]) + (f[..., 1] * x_pairs[..., 1])
    return out.reshape(B, H, N, D)


class EmbedND(nn.Module):
    """3-axis RoPE embedding table (frame, height, width) for Krea2.

    axes_dim=(32, 48, 48) sums to head_dim=128. Produces a rotation-matrix
    table, not separate cos/sin tensors — matches FLUX's own EmbedND.
    """

    def __init__(self, axes_dim: tuple[int, ...], theta: float = 1000.0):
        super().__init__()
        self.axes_dim = axes_dim
        self.theta = theta

    def __call__(self, pos: mx.array) -> mx.array:
        """
        Args:
            pos: [N, num_axes] position indices, one column per axis.

        Returns:
            [N, head_dim/2, 2, 2] rotation matrices, concatenated across axes.
        """
        parts = [_rope_freqs_axis(pos[:, i], self.axes_dim[i], self.theta)
                  for i in range(len(self.axes_dim))]
        return mx.concatenate(parts, axis=-3)


# ── load_krea2_transformer ─────────────────────────────────────────────

def load_krea2_transformer(
    path,
    dtype: str = "float16",
) -> SingleStreamDiT:
    """Load a Krea2 checkpoint into a SingleStreamDiT.

    Args:
        path: Path to the safetensors checkpoint file.
        dtype: Target dtype ("float16" or "bfloat16").

    Returns:
        Loaded SingleStreamDiT instance.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    from .. import _load_safetensors

    # Load state dict
    state_dict = _load_safetensors(path)

    # Normalize and map keys
    from .weight_map import normalize_krea2_keys, map_krea2_to_native
    state_dict = normalize_krea2_keys(state_dict)
    state_dict = map_krea2_to_native(state_dict)

    # Create config and model
    from .config import Krea2Config
    config = Krea2Config(dtype=dtype)
    transformer = SingleStreamDiT(config)

    # Assign weights using tree_unflatten (update() doesn't handle flat string keys
    # for models with list-indexed hierarchies like blocks[0].attn.wq).
    # tree_flatten returns (dotted_string_key, array) pairs — match directly
    # against the checkpoint's own dotted string keys (both use "." and plain
    # integer block indices), no tuple conversion needed.
    model_flat = tree_flatten(transformer.parameters())

    new_flat = []
    matched = 0
    for flat_key, value in model_flat:
        if flat_key in state_dict:
            new_flat.append((flat_key, state_dict[flat_key].astype(config.mlx_dtype)))
            matched += 1
        else:
            # mx.random-initialized params (nn.Linear's default bias is a
            # non-zero uniform draw, not zero -- confirmed on this MLX
            # version) must not leak into inference as leftover training
            # noise when a checkpoint simply doesn't ship that tensor (e.g.
            # a bias-free-trained Turbo variant, which has NO bias.* keys
            # at all for any block -- verified against a real checkpoint
            # that matched exactly 430/686, the missing 256 all being
            # attn/mlp .bias tensors). Zero is the correct "this checkpoint
            # has no such tensor" default for inference, not a random draw.
            new_flat.append((flat_key, mx.zeros_like(value)))
    new_nested = tree_unflatten(new_flat)
    transformer.update(new_nested)
    mx.eval(transformer.parameters())

    print(f"[ASDX] Krea2 transformer: matched {matched}/{len(model_flat)} params from checkpoint")
    return transformer
