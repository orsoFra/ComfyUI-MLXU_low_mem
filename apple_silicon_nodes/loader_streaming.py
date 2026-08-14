"""Low-peak checkpoint loader.

The original loader remains in :mod:`loader`.  This module replaces only the
SDXL checkpoint node with a tensor-at-a-time reader; other model families use
the original implementation unchanged.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from math import prod
from typing import Any

import mlx.core as mx

from . import bridge
from .loader import (
    ASDX_CheckpointLoader as _OriginalCheckpointLoader,
    ASDX_DiffusionLoader,
    _capability_for_model_type,
    _detect_model_type,
    _gate_memory_before_load,
    _purge_stale_asdx_cache_entries,
)
from .native import _check_weight_match
from .native.sdxl.config import SDXLConfig
from .native.sdxl.model import UNetModel
from .native.sdxl.weight_map import native_key_to_checkpoint_stem
from .native.weight_format import QuantFormat, classify_quant_format
from .native.safetensors_header import read_safetensors_header, verify_safetensors_integrity


@dataclass(frozen=True)
class _ShapeOnlyTensor:
    """Enough tensor metadata for ComfyUI's SDXL model-config detection.

    The CLIP/VAE loader is called with ``output_model=False``; the UNet
    entries are consulted only for shape/dtype-based detection, so storing
    their values would be an unnecessary second copy of the whole UNet.
    """
    shape: tuple[int, ...]
    dtype: Any

    def numel(self) -> int:
        return prod(self.shape)

    def nelement(self) -> int:
        return self.numel()


def _torch_dtype(safetensors_dtype: str, torch: Any) -> Any:
    return {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
    }.get(safetensors_dtype, torch.float16)


def _load_embedded_sdxl_clip_vae(path: Path):
    """Load only real SDXL CLIP/VAE tensors from a merged checkpoint.

    ComfyUI's normal checkpoint helper first materializes every tensor,
    including the UNet we already loaded into MLX.  Its SDXL configuration
    detection only needs UNet names, shapes, and dtypes, represented here by
    :class:`_ShapeOnlyTensor` without allocating their payloads.
    """
    import torch
    import comfy.sd
    from safetensors import safe_open

    header = read_safetensors_header(path)
    state: dict[str, Any] = {}
    component_prefixes = ("first_stage_model.", "conditioner.")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith(component_prefixes):
                state[key] = checkpoint.get_tensor(key)
    for key, entry in header.tensors.items():
        if key.startswith("model.diffusion_model."):
            state[key] = _ShapeOnlyTensor(entry.shape, _torch_dtype(entry.dtype, torch))

    component_count = sum(not isinstance(value, _ShapeOnlyTensor) for value in state.values())
    skeleton_count = len(state) - component_count
    print(f"[ASDX][STREAMING-LOADER] selective CLIP/VAE read: "
          f"{component_count} real tensors, {skeleton_count} UNet shape entries")

    _, clip, vae, _ = comfy.sd.load_state_dict_guess_config(
        state, output_model=False, output_clip=True, output_vae=True,
        output_clipvision=False, metadata=header.metadata,
    )
    state.clear()
    gc.collect()
    return clip, vae


def _stream_sdxl_unet(path: Path, dtype: str) -> UNetModel:
    """Load dense SDXL UNet tensors one at a time into MLX.

    Crucially, this never constructs the full checkpoint Torch state dict or
    the full raw MLX state dict.  A source tensor is released immediately
    after its converted MLX parameter has been materialized.
    """
    import torch
    from mlx.utils import tree_flatten, tree_unflatten
    from safetensors import safe_open

    header = read_safetensors_header(path)
    verify_safetensors_integrity(path, header)
    if classify_quant_format(header) != QuantFormat.DENSE:
        raise ValueError("streaming SDXL loader supports dense F16/BF16/F32 checkpoints only")

    config = SDXLConfig(dtype=dtype)
    unet = UNetModel(config)
    model_flat = tree_flatten(unet.parameters())
    total_parameters = len(model_flat)
    checkpoint_keys: set[str]
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
        matched = 0
        for native_key, existing in model_flat:
            stem = native_key_to_checkpoint_stem(native_key)
            source_key = next((prefix + stem for prefix in (
                "model.diffusion_model.", "diffusion_model.", "model.", ""
            ) if prefix + stem in checkpoint_keys), None)
            if source_key is None:
                continue

            source = checkpoint.get_tensor(source_key)
            if source.dtype == torch.bfloat16:
                source = source.float()
            raw = mx.array(source.cpu().numpy())
            del source
            converted = mx.transpose(raw, (0, 2, 3, 1)) if raw.ndim == 4 else raw
            converted = converted.astype(config.mlx_dtype)
            mx.eval(converted)
            del raw
            # `Module.update()` accepts a partial tree.  Replacing each
            # parameter here drops the randomly initialized counterpart
            # before the next checkpoint tensor is converted.
            unet.update(tree_unflatten([(native_key, converted)]))
            del converted
            matched += 1
            if matched % 32 == 0:
                mx.clear_cache()

    # Drop references to the initial lazy parameter arrays before the final
    # model-wide evaluation.
    del model_flat, existing
    mx.eval(unet.parameters())
    gc.collect()
    mx.clear_cache()
    _check_weight_match(matched, total_parameters, "SDXL UNet", path)
    print(f"[ASDX] Streaming SDXL UNet: matched {matched}/{total_parameters} parameters from checkpoint")
    return unet


class ASDX_CheckpointLoader(_OriginalCheckpointLoader):
    """Original checkpoint loader with a low-peak dense-SDXL path."""

    @classmethod
    def execute(cls, ckpt_name: str, precision: str):
        path = cls._resolve_checkpoint_path(ckpt_name)
        model_type = _detect_model_type(path)
        if model_type != "sdxl":
            return _OriginalCheckpointLoader.execute(ckpt_name, precision)

        print(f"[ASDX][STREAMING-LOADER] active for dense SDXL: {ckpt_name}")
        t0 = time.perf_counter()
        purged = _purge_stale_asdx_cache_entries()
        if purged:
            print(f"[ASDX] Purged {purged} stale ComfyUI executor cache entries")
        bridge.clear_mlx_cache()
        memory_shape = _gate_memory_before_load(path, model_type, precision, low_memory_mode=False)
        transformer = _stream_sdxl_unet(path, precision)
        mem_after_unet = bridge.collect_mlx_memory()
        print(f"[ASDX][STREAMING-LOADER] native UNet ready "
              f"({mem_after_unet['active_gb']:.1f}GB active, "
              f"{mem_after_unet['cache_gb']:.1f}GB cache); loading Comfy CLIP/VAE")

        clip, vae = _load_embedded_sdxl_clip_vae(path)
        model_desc = {
            "type": "asdx_model", "name": ckpt_name, "path": str(path),
            "transformer": transformer, "config": SDXLConfig(dtype=precision),
            "model_type": model_type, "precision": precision,
            "capability": _capability_for_model_type(model_type, path),
            "memory_shape": memory_shape,
        }
        mem = bridge.collect_mlx_memory()
        print(f"[ASDX] Streaming checkpoint loaded: {ckpt_name} in {time.perf_counter() - t0:.1f}s "
              f"(mem={mem['active_gb']:.1f}GB active, {mem['cache_gb']:.1f}GB cache)")
        from comfy_api.latest import io
        return io.NodeOutput(model_desc, clip, vae)


NODE_LIST = [ASDX_DiffusionLoader, ASDX_CheckpointLoader]
