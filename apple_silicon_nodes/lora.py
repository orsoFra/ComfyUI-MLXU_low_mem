"""
LoRA Runtime Loading
====================
Load and apply LoRA adapters to FLUX transformers at runtime.

Supports:
  - Standard LoRA (A/B matrices)
  - ComfyUI diff format (.diff, .diff_b)
  - Per-LoRA alpha scaling (alpha / rank)
  - Multiple LoRA stacking with individual weights
  - LoRA baking (fuse into base weights for speed)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch

from . import bridge


# ── LoRA Target Definition ────────────────────────────────────────────

@dataclass(frozen=True)
class LoRATarget:
    """Defines which transformer layer a LoRA weight key maps to."""
    # Path through the transformer module tree
    path_parts: tuple[str, ...]
    # Weight key suffixes in the LoRA file
    key_suffixes: tuple[str, ...]  # e.g. ("lora_A.weight", "lora_B.weight")


# FLUX.1 LoRA target patterns
_FLUX_LORA_TARGETS: tuple[LoRATarget, ...] = (
    # Double block attention
    LoRATarget(("double_blocks", "{i}", "img_attn"), ("qkv.weight",)),
    LoRATarget(("double_blocks", "{i}", "img_attn"), ("proj.weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_attn"), ("qkv.weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_attn"), ("proj.weight",)),
    # Double block MLP
    LoRATarget(("double_blocks", "{i}", "img_mlp_0"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "img_mlp_2"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_mlp_0"), ("weight",)),
    LoRATarget(("double_blocks", "{i}", "txt_mlp_2"), ("weight",)),
    # Single block attention
    LoRATarget(("single_blocks", "{i}", "attn"), ("qkv.weight",)),
    LoRATarget(("single_blocks", "{i}", "attn"), ("proj.weight",)),
    # Single block MLP
    LoRATarget(("single_blocks", "{i}", "mlp_0"), ("weight",)),
    LoRATarget(("single_blocks", "{i}", "mlp_2"), ("weight",)),
)


# ── LoRA Adapter ─────────────────────────────────────────────────────

@dataclass
class LoRAAdapter:
    """A single LoRA adapter with its weights and scale."""
    name: str
    # Map from (block_index, layer_type, param) -> delta weight
    deltas: dict[tuple[int, str, str], mx.array] = field(default_factory=dict)
    alpha: float = 1.0
    rank: int = 0
    scale: float = 1.0

    def __post_init__(self):
        if self.rank == 0:
            # Infer rank from first delta
            if self.deltas:
                self.rank = next(iter(self.deltas.values())).shape[-1]
        if self.scale == 0:
            self.scale = self.alpha / max(self.rank, 1)


# ── LoRA Loader Node ─────────────────────────────────────────────────

class ASDX_LoraLoader:
    """Load a LoRA adapter and apply it to a model.

    Supports standard LoRA (A/B matrices) and ComfyUI diff format.
    Multiple LoRAs can be stacked by chaining LoRA loaders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("asdx_model",),
                "lora_name": (cls._get_loras(),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "alpha": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("asdx_model",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "ASDX/LoRA"

    @staticmethod
    def _get_loras() -> list[str]:
        """Get list of available LoRA files."""
        try:
            import folder_paths
            loras = []
            for folder in ("loras",):
                try:
                    loras.extend(folder_paths.get_filename_list(folder))
                except Exception:
                    pass
            if loras:
                return loras
        except Exception:
            pass
        return ["example_lora.safetensors"]

    def load_lora(
        self,
        model: dict,
        lora_name: str,
        strength_model: float,
        alpha: float,
    ) -> tuple[dict]:
        """Load and apply a LoRA adapter to the model."""
        t0 = time.perf_counter()

        transformer = model["transformer"]
        lora_path = self._resolve_lora_path(lora_name)

        # Load LoRA weights
        lora = self._load_lora_file(lora_path)

        # Apply scale
        lora.scale = alpha / max(lora.rank, 1) * strength_model

        # Apply to transformer
        self._apply_lora_to_transformer(transformer, lora)

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] LoRA '{lora_name}' applied: rank={lora.rank}, "
              f"scale={lora.scale:.4f}, {elapsed:.2f}s")

        return (model,)

    @staticmethod
    def _load_lora_file(path: Path) -> LoRAAdapter:
        """Load a LoRA file and extract delta weights."""
        t0 = time.perf_counter()
        name = path.stem

        if path.suffix == ".safetensors":
            import safetensors
            with open(path, "rb") as f:
                raw = safetensors.numpy.load(f.read())
        elif path.suffix == ".pt" or path.suffix == ".bin":
            import torch
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict):
                raw = {k: v.numpy() for k, v in state.items()}
            else:
                raw = {}
        else:
            raise ValueError(f"Unsupported LoRA format: {path.suffix}")

        # Extract deltas from raw weights
        lora = LoRAAdapter(name=name, alpha=1.0)
        deltas: dict[str, tuple[mx.array, mx.array]] = {}  # key -> (A, B) or diff

        for key, weight in raw.items():
            weight_arr = mx.array(weight if isinstance(weight, mx.array) else weight)

            # Standard LoRA format: {prefix}.lora_A.{param} / {prefix}.lora_B.{param}
            if ".lora_A." in key:
                prefix = key.replace(".lora_A.", ".")
                if prefix not in deltas:
                    deltas[prefix] = (None, None)
                deltas[prefix] = (weight_arr, deltas[prefix][1])
            elif ".lora_B." in key:
                prefix = key.replace(".lora_B.", ".")
                if prefix not in deltas:
                    deltas[prefix] = (deltas[prefix][0], None)
                deltas[prefix] = (deltas[prefix][0], weight_arr)
            elif ".lora_up." in key:
                prefix = key.replace(".lora_up.", ".")
                up_key = prefix.replace(".lora_up.", ".lora_up.")
                down_key = prefix.replace(".lora_up.", ".lora_down.")
                if up_key in raw and down_key in raw:
                    if isinstance(raw[down_key], mx.array):
                        lora.deltas[prefix] = (raw[down_key], raw[up_key])
                    else:
                        lora.deltas[prefix] = (mx.array(raw[down_key]), mx.array(raw[up_key]))
            elif ".diff" in key:
                # ComfyUI diff format
                diff_key = key.replace(".diff", "")
                if ".diff_b" in key:
                    lora.deltas[diff_key] = (None, weight_arr)
                else:
                    lora.deltas[diff_key] = (weight_arr, None)

        # Convert (A, B) pairs to delta = B @ A and compute rank
        for key, (a, b) in lora.deltas.items():
            if a is not None and b is not None:
                # Standard LoRA: delta = B @ A
                delta = (b.astype(mx.float32) @ a.astype(mx.float32)).astype(b.dtype)
                lora.deltas[key] = delta
                if lora.rank == 0:
                    lora.rank = a.shape[-1]
            elif a is not None:
                lora.deltas[key] = a
                if lora.rank == 0:
                    lora.rank = a.shape[-1]
            elif b is not None:
                lora.deltas[key] = b
                if lora.rank == 0:
                    lora.rank = b.shape[-1]

        # Clean up unused pairs
        lora.deltas = {k: v for k, v in lora.deltas.items()
                       if not (isinstance(v, tuple) and v[0] is None and v[1] is None)}

        mx.eval(*lora.deltas.values())
        return lora

    @staticmethod
    def _apply_lora_to_transformer(
        transformer: Any,
        lora: LoRAAdapter,
    ) -> None:
        """Apply LoRA delta weights to transformer parameters.

        LoRA is applied by adding scaled delta weights to the base weights.
        This modifies the transformer in-place.
        """
        # Build a mapping from weight keys to delta weights
        delta_map = {}
        for key, delta in lora.deltas.items():
            # key could be something like "double_blocks.0.img_attn.qkv.weight"
            # or "single_blocks.5.mlp_0.weight"
            delta_map[key] = delta

        if not delta_map:
            print("[ASDX] LoRA: no matching weights found")
            return

        # Apply deltas to transformer weights
        applied = 0
        for key, delta in delta_map.items():
            # Navigate to the parameter in the transformer
            param = transformer
            parts = key.split(".")
            for i, part in enumerate(parts):
                # Handle numeric indices (block numbers)
                if part.isdigit():
                    idx = int(part)
                    if hasattr(param, "__iter__") and not isinstance(param, str):
                        param = list(param)[idx]
                    else:
                        param = param[idx]
                    continue

                # Handle attribute access
                if hasattr(param, part):
                    param = getattr(param, part)
                elif hasattr(param, "_parameters") and part in param._parameters:
                    param = param._parameters[part]
                else:
                    param = None
                    break

            if param is not None and hasattr(param, "value"):
                # MLX nn.Parameter
                old = param.value
                delta_mapped = delta.astype(old.dtype)
                param.value = old + delta_mapped * lora.scale
                applied += 1
            elif hasattr(param, "weight") and param.weight is not None:
                # nn.Linear layer
                old = param.weight
                delta_mapped = delta.astype(old.dtype)
                param.weight = old + delta_mapped * lora.scale
                applied += 1

        mx.eval(transformer.parameters())
        print(f"[ASDX] LoRA: applied {applied}/{len(delta_map)} deltas")

    @staticmethod
    def _resolve_lora_path(name: str) -> Path:
        """Resolve LoRA name to file path."""
        try:
            import folder_paths
            for folder in ("loras",):
                try:
                    full = folder_paths.get_full_path(folder, name)
                    if full:
                        return Path(full)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback
        for candidate in (
            Path.home() / "ComfyUI" / "models" / "loras" / name,
            Path(name),
        ):
            if candidate.exists():
                return candidate
        return Path(name)


# ── Multi LoRA Loader ────────────────────────────────────────────────

class ASDX_MultiLoraLoader:
    """Load multiple LoRA adapters with individual strengths.

    Applies all LoRAs in a single pass for better performance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("asdx_model",),
                "lora1_name": (ASDX_LoraLoader._get_loras(),),
                "lora1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora2_name": (ASDX_LoraLoader._get_loras(),),
                "lora2_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora3_name": (ASDX_LoraLoader._get_loras(),),
                "lora3_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora4_name": (ASDX_LoraLoader._get_loras(),),
                "lora4_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora5_name": (ASDX_LoraLoader._get_loras(),),
                "lora5_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("asdx_model",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_loras"
    CATEGORY = "ASDX/LoRA"

    def load_loras(
        self,
        model: dict,
        lora1_name: str, lora1_strength: float,
        lora2_name: str, lora2_strength: float,
        lora3_name: str, lora3_strength: float,
        lora4_name: str, lora4_strength: float,
        lora5_name: str, lora5_strength: float,
    ) -> tuple[dict]:
        """Apply up to 5 LoRA adapters at once."""
        loras = [
            (lora1_name, lora1_strength),
            (lora2_name, lora2_strength),
            (lora3_name, lora3_strength),
            (lora4_name, lora4_strength),
            (lora5_name, lora5_strength),
        ]

        transformer = model["transformer"]
        applied_any = False

        for lora_name, strength in loras:
            if not lora_name or lora_name == "None" or strength == 0:
                continue

            lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
            lora = ASDX_LoraLoader._load_lora_file(lora_path)
            lora.scale = lora.alpha * strength

            # Merge deltas into a single pass
            self._apply_single_lora(transformer, lora)
            applied_any = True
            print(f"[ASDX] MultiLoRA: '{lora_name}' (strength={strength:.2f})")

        if not applied_any:
            print("[ASDX] MultiLoRA: no LoRAs to apply")

        return (model,)

    @staticmethod
    def _apply_single_lora(transformer: Any, lora: LoRAAdapter) -> None:
        """Apply a single LoRA to transformer (same as above but without logging)."""
        for key, delta in lora.deltas.items():
            param = transformer
            parts = key.split(".")
            for part in parts:
                if part.isdigit():
                    idx = int(part)
                    if hasattr(param, "__iter__") and not isinstance(param, str):
                        param = list(param)[idx]
                    else:
                        param = param[idx]
                    continue
                if hasattr(param, part):
                    param = getattr(param, part)
                elif hasattr(param, "_parameters") and part in param._parameters:
                    param = param._parameters[part]
                else:
                    param = None
                    break

            if param is not None:
                if hasattr(param, "value"):
                    old = param.value
                    param.value = old + delta.astype(old.dtype) * lora.scale
                elif hasattr(param, "weight") and param.weight is not None:
                    old = param.weight
                    param.weight = old + delta.astype(old.dtype) * lora.scale

        mx.eval(transformer.parameters())


# ── LoRA Schedule (per-step strength modulation) ─────────────────────

class ASDX_LoraSchedule:
    """Schedule LoRA strength across sampling steps.

    Allows LoRA strength to vary per step (e.g., stronger at start, weaker at end).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("asdx_model",),
                "lora_name": (ASDX_LoraLoader._get_loras(),),
                "strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_end": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_middle": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "strength_curve": (["linear", "cosine", "ease_in_out"],),
            },
        }

    RETURN_TYPES = ("asdx_model",)
    RETURN_NAMES = ("model",)
    FUNCTION = "schedule_lora"
    CATEGORY = "ASDX/Advanced"

    def schedule_lora(
        self,
        model: dict,
        lora_name: str,
        strength_start: float,
        strength_end: float,
        strength_middle: float,
        strength_curve: str,
    ) -> tuple[dict]:
        """Attach LoRA schedule metadata to the model dict.

        The sampler reads this metadata and adjusts LoRA strength per step.
        """
        lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
        lora = ASDX_LoraLoader._load_lora_file(lora_path)

        # Store schedule info on the model
        model["lora_schedule"] = {
            "name": lora_name,
            "lora": lora,
            "strength_start": strength_start,
            "strength_end": strength_end,
            "strength_middle": strength_middle,
            "strength_curve": strength_curve,
        }

        # Apply with start strength
        lora.scale = lora.alpha * strength_start
        ASDX_LoraLoader._apply_lora_to_transformer(model["transformer"], lora)

        print(f"[ASDX] LoRA schedule: '{lora_name}' "
              f"start={strength_start:.2f} middle={strength_middle:.2f} end={strength_end:.2f}")

        return (model,)

    @staticmethod
    def _compute_schedule_strength(
        step: int,
        total_steps: int,
        start: float,
        end: float,
        middle: float,
        curve: str,
    ) -> float:
        """Compute LoRA strength for a given step."""
        if total_steps <= 1:
            return start

        progress = step / total_steps

        if curve == "linear":
            # Start -> Middle (0-0.5) -> End (0.5-1.0)
            if progress <= 0.5:
                return start + (middle - start) * (progress * 2)
            else:
                return middle + (end - middle) * ((progress - 0.5) * 2)

        elif curve == "cosine":
            # Smooth cosine interpolation
            mid = (start + middle) / 2
            end_val = (middle + end) / 2
            if progress <= 0.5:
                return mid + (start - mid) * 0.5 * (1 - mx.cos(mx.array(progress * 2 * 3.14159)))
            else:
                return end_val + (end - end_val) * 0.5 * (1 - mx.cos(mx.array((progress - 0.5) * 2 * 3.14159)))

        elif curve == "ease_in_out":
            # S-curve: slow start, fast middle, slow end
            t = 3 * progress ** 2 - 2 * progress ** 3
            if progress <= 0.5:
                return start + (middle - start) * t
            else:
                return middle + (end - middle) * t

        return start  # default: constant


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ASDX_LoraLoader": ASDX_LoraLoader,
    "ASDX_MultiLoraLoader": ASDX_MultiLoraLoader,
    "ASDX_LoraSchedule": ASDX_LoraSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_LoraLoader": "🍏 ASDX LoRA Loader",
    "ASDX_MultiLoraLoader": "🍏 ASDX Multi LoRA Loader",
    "ASDX_LoraSchedule": "🍏 ASDX LoRA Schedule",
}
