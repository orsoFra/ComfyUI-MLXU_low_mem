"""
ControlNet Union Support
========================
MLX-native ControlNet Union ProMax for FLUX.

Supports 8 control types:
  pose, depth, soft_edge, line_canny, normal, segment, tile, repaint
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from . import bridge


# ── ControlNet Types ─────────────────────────────────────────────────

CONTROL_NET_TYPES = {
    "pose": 0,
    "depth": 1,
    "soft_edge": 2,
    "line_canny": 3,
    "normal": 4,
    "segment": 5,
    "tile": 6,
    "repaint": 7,
}


# ── ControlNet Building Blocks ───────────────────────────────────────

class ControlNetCondEmbedding(nn.Module):
    """Embed control image into conditioning features."""

    def __init__(self, conditioning_channels=3, block_out_channels=None,
                 embedding_channels=320):
        super().__init__()
        if block_out_channels is None:
            block_out_channels = [32, 64, 128, 256]
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0],
                                 kernel_size=3, padding=1)
        self.blocks = []
        for i in range(len(block_out_channels) - 1):
            ch_in = block_out_channels[i]
            ch_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(ch_in, ch_out, kernel_size=3,
                                         stride=2, padding=1))
        self.conv_out = nn.Conv2d(block_out_channels[-1], embedding_channels,
                                  kernel_size=3, padding=1)

    def __call__(self, conditioning):
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


# ── ControlNet Union Model ───────────────────────────────────────────

class ControlNetUnionModel(nn.Module):
    """ControlNet Union ProMax supporting 8 control types.

    Architecture:
      - conv_in: Conv2d(4, 320) -- accepts RGB + mask
      - controlnet_cond_embedding: processes control image
      - control_type_proj: encodes which control type is active
      - transformer_layers: 1 ResidualAttentionBlock(320, 8)
      - down_blocks: 3 UNetBlock2D stages [320, 640, 1280]
      - mid_block: ResNet -> Transformer2D(20 heads, 10 layers) -> ResNet
      - controlnet_down_blocks: 9 residual projections (8 down + 1 mid)
    """

    def __init__(self, num_control_types: int = 8):
        super().__init__()
        self.num_control_type = num_control_types

        # Timestep encoding
        self.timesteps = SinusoidalPositionalEncoding(320, scale=1.0, cos_first=True)
        self.time_embedding = TimeEmbedding(320, 1280)

        # Control type embeddings
        self.control_type_proj = SinusoidalPositionalEncoding(256, scale=1.0, cos_first=True)
        self.control_add_embedding = TimeEmbedding(256 * num_control_types, 1280)

        # Control image conditioning
        self.conv_in = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.controlnet_cond_embedding = ControlNetCondEmbedding()
        self.task_embedding = mx.zeros((num_control_types, 320))
        self.transformer_layers = [self._make_attention_block(320, 8)]
        self.spatial_ch_projs = nn.Linear(320, 320)

        # Down blocks (simplified UNet structure)
        self.down_blocks = [
            self._make_unet_block(320, 320, temb_channels=1280, add_cross_attention=False),
            self._make_unet_block(320, 640, temb_channels=1280, add_cross_attention=True),
            self._make_unet_block(640, 1280, temb_channels=1280, add_cross_attention=True),
        ]

        # Mid block
        self.mid_blocks = [
            self._make_resnet(1280, 1280, temb_channels=1280),
            self._make_transformer(1280, 2048, 20, 10),
            self._make_resnet(1280, 1280, temb_channels=1280),
        ]

        # Output projections
        control_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]
        self.controlnet_down_blocks = [
            nn.Conv2d(ch, ch, kernel_size=1) for ch in control_channels
        ]
        self.controlnet_mid_block = nn.Conv2d(1280, 1280, kernel_size=1)

    def _make_attention_block(self, dim: int, num_heads: int) -> nn.Module:
        """Create a ResidualAttentionBlock."""
        return nn.Sequential([
            nn.LayerNorm(dim),
            nn.MultiHeadAttention(dim, num_heads, bias=True),
            nn.LayerNorm(dim),
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                lambda x: x * mx.sigmoid(1.702 * x),
                nn.Linear(dim * 4, dim),
            ),
        ])

    def _make_unet_block(self, in_ch: int, out_ch: int,
                         temb_channels: int, add_cross_attention: bool) -> nn.Module:
        """Create a simplified UNet block."""
        return nn.Sequential([
            self._make_resnet(in_ch, out_ch, temb_channels),
            self._make_resnet(out_ch, out_ch, temb_channels),
        ])

    def _make_resnet(self, in_ch: int, out_ch: int,
                     temb_channels: int) -> nn.Module:
        """Create a ResNet block."""
        return nn.Sequential([
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            TimeEmbedding(temb_channels, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        ])

    def _make_transformer(self, dim: int, cross_attn_dim: int,
                          num_heads: int, num_layers: int) -> nn.Module:
        """Create a transformer block."""
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.LayerNorm(dim),
                nn.MultiHeadAttention(dim, num_heads, bias=True),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                lambda x: x * mx.sigmoid(1.702 * x),
                nn.Linear(dim * 4, dim),
            ])
        return nn.Sequential(layers)

    def __call__(
        self,
        x: mx.array,           # [B, C, H, W] noisy latent
        timestep: mx.array,    # [B] timestep
        encoder_x: mx.array,   # [B, T, D] text embeddings
        control_image: mx.array,  # [B, 4, H, W] control image (RGB + mask)
        control_type_idx: int,  # Which control type
        conditioning_scale: float = 1.0,
        text_time: tuple | None = None,
    ) -> tuple[list[mx.array], mx.array]:
        """Forward pass.

        Returns:
            (down_residuals, mid_residual) each scaled by conditioning_scale
        """
        dtype = x.dtype
        batch = x.shape[0]

        # Timestep embedding
        temb = self.time_embedding(self.timesteps(timestep).astype(dtype))

        # Add text time
        if text_time is not None:
            text_emb, time_ids = text_time
            emb = self._time_proj(time_ids).flatten(1).astype(dtype)
            emb = mx.concatenate([text_emb, emb], axis=-1)
            temb = temb + self.control_add_embedding(emb)

        # Control type embedding
        control_type = mx.zeros((batch, self.num_control_type), dtype=dtype)
        control_type = control_type + mx.array(
            [[1.0 if i == control_type_idx else 0.0 for i in range(self.num_control_type)]],
            dtype=dtype,
        )
        control_embeds = self.control_type_proj(control_type.flatten()).reshape(batch, -1).astype(dtype)
        temb = temb + self.control_add_embedding(control_embeds)

        # Process control image
        x = self.conv_in(x)
        condition = self.controlnet_cond_embedding(control_image.astype(dtype))
        feat_seq = mx.mean(condition, axis=(1, 2)) + self.task_embedding[control_type_idx].astype(dtype)
        sample_seq = mx.mean(x, axis=(1, 2))
        seq = mx.stack([feat_seq, sample_seq], axis=1)

        for layer in self.transformer_layers:
            seq = layer(seq)

        alpha = self.spatial_ch_projs(seq[:, 0])[:, None, None, :]
        x = x + condition + alpha

        # Down blocks
        residuals = [x]
        for block in self.down_blocks:
            x = block[0](x, temb)
            x = block[1](x, temb)
            residuals.append(x)

        # Mid block
        x = self.mid_blocks[0](x[0], temb)
        x = self.mid_blocks[1](x, encoder_x, None, None)
        x = self.mid_blocks[2](x, temb)

        # Scale residuals
        down = [
            block(residual) * conditioning_scale
            for block, residual in zip(self.controlnet_down_blocks, residuals)
        ]
        mid = self.controlnet_mid_block(x) * conditioning_scale

        return down, mid

    def _time_proj(self, time_ids: mx.array) -> mx.array:
        """Timestep projection for text_time augmentation."""
        half_dim = 256 // 2
        exponent = -math.log(10000) * mx.arange(start=0, stop=half_dim, dtype=mx.float32) / half_dim
        emb = mx.exp(exponent)
        emb = time_ids[:, None].astype(mx.float32) * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


# ── Model Loader ─────────────────────────────────────────────────────

_CONTROLNET_CACHE: dict[str, ControlNetUnionModel] = {}


def load_controlnet_union(path: str | Path) -> ControlNetUnionModel:
    """Load a ControlNet Union model from a checkpoint file."""
    path = Path(path)
    cache_key = str(path)

    if cache_key in _CONTROLNET_CACHE:
        return _CONTROLNET_CACHE[cache_key]

    # Load weights
    import safetensors
    with open(path, "rb") as f:
        raw = safetensors.numpy.load(f.read())

    # Create model
    model = ControlNetUnionModel()

    # Map weights
    _assign_controlnet_weights(model, raw)

    mx.eval(model.parameters())
    _CONTROLNET_CACHE[cache_key] = model

    print(f"[ASDX] ControlNet Union loaded: {path.name}")
    return model


def _assign_controlnet_weights(model: ControlNetUnionModel, raw: dict) -> None:
    """Assign weights from checkpoint to model parameters."""
    for key, value in raw.items():
        weight = mx.array(value)

        # Normalize key
        prefix = "control_model."
        if key.startswith(prefix):
            key = key[len(prefix):]

        # Map to model attributes
        _set_param(model, key, weight)


def _set_param(obj: Any, key: str, value: mx.array) -> None:
    """Set a parameter on an MLX module, handling nested paths."""
    parts = key.split(".")

    # Handle special key transformations
    if key.endswith(".attn.in_proj_weight"):
        prefix = key[:-len(".attn.in_proj_weight")]
        q, k, v = mx.split(value, 3)
        _set_param(obj, f"{prefix}.attn.query_proj.weight", q)
        _set_param(obj, f"{prefix}.attn.key_proj.weight", k)
        _set_param(obj, f"{prefix}.attn.value_proj.weight", v)
        return
    if key.endswith(".attn.in_proj_bias"):
        prefix = key[:-len(".attn.in_proj_bias")]
        q, k, v = mx.split(value, 3)
        _set_param(obj, f"{prefix}.attn.query_proj.bias", q)
        _set_param(obj, f"{prefix}.attn.key_proj.bias", k)
        _set_param(obj, f"{prefix}.attn.value_proj.bias", v)
        return

    # Handle ln_1 -> norm1, ln_2 -> norm2
    key = key.replace(".ln_1.", ".norm1.")
    key = key.replace(".ln_2.", ".norm2.")
    key = key.replace(".transofrmer_layes.", ".transformer_layers.")
    key = key.replace(".transformer_layes.", ".transformer_layers.")

    # Navigate to parent object
    current = obj
    for part in parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "_parameters") and part in current._parameters:
            current = current._parameters[part]
        else:
            return  # Key not found

    # Set the final parameter
    attr = parts[-1]
    if hasattr(current, attr):
        current_val = getattr(current, attr)
        if hasattr(current_val, "shape") and current_val.shape == value.shape:
            setattr(current, attr, value)
    elif hasattr(current, "_parameters") and attr in current._parameters:
        if current._parameters[attr].shape == value.shape:
            current._parameters[attr] = value


# ── Nodes ─────────────────────────────────────────────────────────────

class ASDX_ControlNetUnionLoader:
    """Load a ControlNet Union model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net_name": (cls._get_controlnets(),),
            },
        }

    RETURN_TYPES = ("controlnet",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "load"
    CATEGORY = "ASDX/ControlNet"

    @staticmethod
    def _get_controlnets() -> list[str]:
        """Get list of available ControlNet models."""
        try:
            import folder_paths
            cns = []
            for folder in ("controlnet",):
                try:
                    cns.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if cns:
                return cns
        except Exception:
            pass
        return ["controlnet_union.safetensors"]

    def load(self, control_net_name: str) -> tuple[dict]:
        path = self._resolve_path(control_net_name)
        control_net = load_controlnet_union(path)
        return ({"control_net": control_net, "name": control_net_name},)

    @staticmethod
    def _resolve_path(name: str) -> Path:
        try:
            import folder_paths
            for folder in ("controlnet",):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "controlnet" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


class ASDX_ApplyControlNet:
    """Apply ControlNet conditioning to a diffusion model.

    Injects ControlNet residuals into the transformer's attention layers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("asdx_model",),
                "control_net": ("controlnet",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "control_type": (["pose", "depth", "soft_edge", "line_canny", "normal", "segment", "tile", "repaint"],),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
            },
        }

    RETURN_TYPES = ("asdx_model",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "ASDX/ControlNet"

    def apply(
        self,
        model: dict,
        control_net: dict,
        image: torch.Tensor,
        strength: float,
        control_type: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[dict]:
        """Attach ControlNet conditioning to the model."""
        type_idx = CONTROL_NET_TYPES.get(control_type, 0)

        # Prepare control image: [B, H, W, 3] -> [B, 4, H, W]
        control_img = self._prepare_control_image(image, mask)

        # Store ControlNet info on the model
        model["controlnet"] = {
            "control_net": control_net["control_net"],
            "image": control_img,
            "strength": strength,
            "control_type": type_idx,
        }

        print(f"[ASDX] ControlNet applied: {control_type} strength={strength:.2f}")
        return (model,)

    @staticmethod
    def _prepare_control_image(image: torch.Tensor, mask: torch.Tensor | None) -> mx.array:
        """Convert image (+ optional mask) to MLX array [B, 4, H, W]."""
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Normalize to [-1, 1]
        img_np = image.detach().cpu().numpy().astype(np.float32)
        img_np = (img_np * 2.0) - 1.0

        # Add mask channel if provided
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            # Transpose mask to [B, 1, H, W]
            if mask_np.ndim == 3:
                mask_np = mask_np[:, None, :, :]
            control = np.concatenate([img_np, mask_np], axis=1)
        else:
            # 3 channels RGB -> add zeros for alpha
            control = np.concatenate([img_np, np.zeros_like(img_np[:, :1])], axis=1)

        control_mlx = mx.array(control)
        mx.eval(control_mlx)
        return control_mlx


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ASDX_ControlNetUnionLoader": ASDX_ControlNetUnionLoader,
    "ASDX_ApplyControlNet": ASDX_ApplyControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_ControlNetUnionLoader": "🍏 ASDX ControlNet Union Loader",
    "ASDX_ApplyControlNet": "🍏 ASDX Apply ControlNet",
}
