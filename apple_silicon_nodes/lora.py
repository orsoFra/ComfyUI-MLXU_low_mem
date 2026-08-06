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

# Krea2 (SingleStreamDiT) LoRA target patterns
# Keys in checkpoint: diffusion_model.blocks.{i}.attn.wq.lora_A.weight
# After stripping prefix: blocks.{i}.attn.wq
_KREA2_LORA_TARGETS: tuple[LoRATarget, ...] = (
    # Attention projections
    LoRATarget(("blocks", "{i}", "attn", "wq"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wk"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wv"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "wo"), ("weight",)),
    LoRATarget(("blocks", "{i}", "attn", "gate_proj"), ("weight",)),
    # MLP projections
    LoRATarget(("blocks", "{i}", "mlp", "up"), ("weight",)),
    LoRATarget(("blocks", "{i}", "mlp", "gate"), ("weight",)),
    LoRATarget(("blocks", "{i}", "mlp", "down"), ("weight",)),
)


# ── LoRA Adapter ─────────────────────────────────────────────────────

@dataclass
class LoRAAdapter:
    """A single LoRA adapter with its weights and scale."""
    name: str
    # Map from (block_index, layer_type, param) -> delta weight
    deltas: dict[tuple[int, str, str], mx.array] = field(default_factory=dict)
    # None means the file has no ".alpha" key (see _load_lora_file) -- not
    # the same as alpha=1.0, the two fall back to different scales below.
    alpha: float | None = None
    rank: int = 0
    scale: float = 1.0

    def __post_init__(self):
        if self.rank == 0:
            # Infer rank from first delta
            if self.deltas:
                self.rank = next(iter(self.deltas.values())).shape[-1]
        if self.scale == 0:
            self.scale = base_lora_scale(self.alpha, self.rank)


def base_lora_scale(alpha: float | None, rank: int) -> float:
    """`alpha/rank` if the file declared its own alpha, else a flat 1.0 --
    matches comfy/weight_adapter/lora.py's fallback exactly (`alpha =
    v[2]/mat2.shape[0] if v[2] is not None else 1.0`). Do NOT default missing
    alpha to `1.0/rank` -- that under-applies the LoRA for any file that
    simply doesn't ship an alpha key.
    """
    return alpha / max(rank, 1) if alpha is not None else 1.0


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
    ) -> tuple[dict]:
        """Load and apply a LoRA adapter to the model."""
        t0 = time.perf_counter()

        transformer = model["transformer"]
        lora_path = self._resolve_lora_path(lora_name)

        # Load LoRA weights (alpha comes from the file itself, see
        # _load_lora_file -- matches real ComfyUI's LoraLoader, which has no
        # user-facing alpha widget either).
        lora = self._load_lora_file(lora_path)

        # Apply scale
        lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_model

        # Apply to transformer — returns a NEW transformer, the cached base
        # model (model["transformer"]) is never mutated (see
        # _apply_lora_to_transformer docstring).
        new_transformer = self._apply_lora_to_transformer(transformer, lora, model["config"])
        new_model = {**model, "transformer": new_transformer}

        # The old (now-orphaned) transformer's arrays are unreferenced after
        # this reassignment, but MLX's allocator won't return that memory to
        # the OS on its own -- trim it now instead of letting it sit as idle
        # cache through however many more nodes run before something else
        # happens to clear it (loader.py does the same after every load).
        bridge.clear_mlx_cache()

        elapsed = time.perf_counter() - t0
        print(f"[ASDX] LoRA '{lora_name}' applied: rank={lora.rank}, "
              f"scale={lora.scale:.4f}, {elapsed:.2f}s")

        return (new_model,)

    @staticmethod
    def _load_lora_file(path: Path) -> LoRAAdapter:
        """Load a LoRA file and extract delta weights."""
        t0 = time.perf_counter()
        name = path.stem

        if path.suffix == ".safetensors":
            import torch
            import safetensors.torch
            state = safetensors.torch.load_file(path, device="cpu")
            raw = {}
            for k, v in state.items():
                if v.dtype == torch.bfloat16:
                    v = v.float()
                raw[k] = v.cpu().numpy()
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
        lora = LoRAAdapter(name=name)
        deltas: dict[str, tuple[mx.array, mx.array]] = {}  # key -> (A, B) or diff
        alpha_value: float | None = None

        for key, weight in raw.items():
            if key.endswith(".alpha"):
                # Real per-file training alpha (comfy/lora.py's convention:
                # alpha_name = "{x}.alpha", same prefix as lora_A/B/up/down
                # minus any .weight suffix). Every target in a real LoRA
                # file shares the same value in practice -- keep the first
                # one found rather than tracking one per target.
                if alpha_value is None:
                    try:
                        alpha_value = float(weight)
                    except (TypeError, ValueError):
                        pass
                continue

            weight_arr = mx.array(weight if isinstance(weight, mx.array) else weight)

            # Standard LoRA format: {prefix}.lora_A.{param} / {prefix}.lora_B.{param}
            if ".lora_A." in key:
                prefix = key.replace(".lora_A.", ".")
                # Strip common ComfyUI prefixes (diffusion_model., model., etc.)
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                # Krea2 checkpoints rename attn.gate -> attn.gate_proj (see
                # weight_map.map_krea2_to_native); LoRA files trained against the
                # checkpoint naming need the same rename to match the live module tree.
                if ".attn.gate.weight" in prefix:
                    prefix = prefix.replace(".attn.gate.", ".attn.gate_proj.")
                if prefix not in deltas:
                    deltas[prefix] = (None, None)
                deltas[prefix] = (weight_arr, deltas[prefix][1])
            elif ".lora_B." in key:
                prefix = key.replace(".lora_B.", ".")
                # Strip common ComfyUI prefixes
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                if ".attn.gate.weight" in prefix:
                    prefix = prefix.replace(".attn.gate.", ".attn.gate_proj.")
                if prefix not in deltas:
                    deltas[prefix] = (deltas[prefix][0], None)
                deltas[prefix] = (deltas[prefix][0], weight_arr)
            elif ".lora_up." in key:
                # kohya-style format: {prefix}.lora_up.weight / {prefix}.lora_down.weight
                down_key = key.replace(".lora_up.", ".lora_down.")
                if down_key not in raw:
                    continue
                prefix = key.replace(".lora_up.", ".")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if prefix.startswith(pfx):
                        prefix = prefix[len(pfx):]
                        break
                if ".attn.gate.weight" in prefix:
                    prefix = prefix.replace(".attn.gate.", ".attn.gate_proj.")
                down_raw = raw[down_key]
                down_arr = down_raw if isinstance(down_raw, mx.array) else mx.array(down_raw)
                # Route through the same (a, b) pairing dict as lora_A/lora_B so the
                # shared "delta = b @ a" conversion loop below handles it uniformly
                # (up=B [out,rank], down=A [rank,in] -- comfy/weight_adapter/lora.py
                # computes `diff = lora_up.weight @ lora_down.weight`, i.e. B @ A).
                deltas[prefix] = (down_arr, weight_arr)
            elif ".diff_b" in key:
                # ComfyUI diff format (bias delta). Real key is "{x}.diff_b" with
                # NO .weight/.bias suffix on x (comfy/lora.py maps it to
                # "{x}.bias") -- append .bias, don't just strip the suffix.
                diff_key = key.replace(".diff_b", ".bias")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if diff_key.startswith(pfx):
                        diff_key = diff_key[len(pfx):]
                        break
                if ".attn.gate.bias" in diff_key:
                    diff_key = diff_key.replace(".attn.gate.", ".attn.gate_proj.")
                lora.deltas[diff_key] = weight_arr
            elif ".diff" in key:
                # ComfyUI diff format (weight delta). Real key is "{x}.diff" with
                # NO .weight suffix on x (comfy/lora.py maps it to "{x}.weight")
                # -- append .weight, don't just strip the suffix.
                diff_key = key.replace(".diff", ".weight")
                for pfx in ("diffusion_model.", "model.", "transformer."):
                    if diff_key.startswith(pfx):
                        diff_key = diff_key[len(pfx):]
                        break
                if ".attn.gate.weight" in diff_key:
                    diff_key = diff_key.replace(".attn.gate.", ".attn.gate_proj.")
                lora.deltas[diff_key] = weight_arr

        # Convert (A, B) pairs to delta = B @ A and compute rank
        # lora_A is [rank, in_features], lora_B is [out_features, rank]
        for key, (a, b) in deltas.items():
            if a is not None and b is not None:
                # Standard LoRA: delta = B @ A
                delta = (b.astype(mx.float32) @ a.astype(mx.float32)).astype(b.dtype)
                lora.deltas[key] = delta
                if lora.rank == 0:
                    lora.rank = a.shape[0]
            elif a is not None:
                lora.deltas[key] = a
                if lora.rank == 0:
                    lora.rank = a.shape[0]
            elif b is not None:
                lora.deltas[key] = b
                if lora.rank == 0:
                    lora.rank = b.shape[-1]

        # Clean up unused pairs
        lora.deltas = {k: v for k, v in lora.deltas.items()
                       if not (isinstance(v, tuple) and v[0] is None and v[1] is None)}

        lora.alpha = alpha_value
        mx.eval(*lora.deltas.values())
        return lora

    @staticmethod
    def _apply_lora_to_transformer(
        transformer: Any,
        lora: LoRAAdapter,
        config: Any,
    ) -> Any:
        """Apply LoRA delta weights to transformer parameters and return a
        NEW transformer instance — never mutates `transformer` in place.

        `transformer` may be the same object cached in loader.py's
        `_MODEL_CACHE` and shared across separate ComfyUI executions; an
        in-place `setattr` here would permanently bake the LoRA into that
        cached base model, so a later cache hit (e.g. re-running the same
        workflow after only changing an unrelated downstream widget) would
        re-apply the delta on top of an already-modified transformer and
        silently compound the LoRA's effect. Every sibling MLX project
        checked (mflux, SDMLX) applies LoRA non-destructively for this same
        reason.

        Untouched parameters are carried over by reference (no copy), so
        this is cheap relative to a real checkpoint reload — only the
        LoRA-targeted arrays are freshly computed.
        """
        from mlx.utils import tree_flatten, tree_unflatten

        if not lora.deltas:
            print("[ASDX] LoRA: no matching weights found")
            return transformer

        # key could be something like "double_blocks.0.img_attn.qkv.weight"
        # or "single_blocks.5.mlp_0.weight" — same dotted-string convention
        # tree_flatten uses for checkpoint keys (see native/*/model.py's
        # load_*_transformer), so deltas match model_flat keys directly.
        model_flat = tree_flatten(transformer.parameters())
        new_flat = []
        applied = 0
        for flat_key, value in model_flat:
            delta = lora.deltas.get(flat_key)
            if delta is not None:
                delta_mapped = delta.astype(value.dtype)
                new_flat.append((flat_key, value + delta_mapped * lora.scale))
                applied += 1
            else:
                new_flat.append((flat_key, value))

        new_transformer = type(transformer)(config)
        new_transformer.update(tree_unflatten(new_flat))
        mx.eval(new_transformer.parameters())
        print(f"[ASDX] LoRA: applied {applied}/{len(lora.deltas)} deltas")
        return new_transformer

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
        config = model["config"]
        applied_any = False

        for lora_name, strength in loras:
            if not lora_name or lora_name == "None" or strength == 0:
                continue

            lora_path = ASDX_LoraLoader._resolve_lora_path(lora_name)
            lora = ASDX_LoraLoader._load_lora_file(lora_path)
            # Same alpha/rank normalization as ASDX_LoraLoader.load_lora, so a
            # rank-N LoRA doesn't apply at N times its intended strength.
            lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength

            # Reuse the (non-destructive, tree-flatten-based) apply logic —
            # do not duplicate it here. Each call returns a NEW transformer;
            # chaining `transformer =` here stacks LoRAs onto that new
            # object, never onto the cached base model.
            transformer = ASDX_LoraLoader._apply_lora_to_transformer(transformer, lora, config)
            # Each iteration orphans the previous intermediate transformer
            # (up to 5 full-size copies chained here) -- trim MLX's idle
            # buffer cache between iterations instead of letting them stack.
            bridge.clear_mlx_cache()
            applied_any = True
            print(f"[ASDX] MultiLoRA: '{lora_name}' (strength={strength:.2f})")

        if not applied_any:
            print("[ASDX] MultiLoRA: no LoRAs to apply")
            return (model,)

        return ({**model, "transformer": transformer},)


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

        # Apply with start strength (same alpha/rank normalization as
        # ASDX_LoraLoader.load_lora — a rank-N LoRA must not apply at N times
        # its intended strength). Returns a NEW transformer — the cached
        # base model is never mutated. `sampler/core.py::_update_lora_schedule`
        # then keeps adjusting THIS new (private, non-cached) transformer
        # in place per step, which is safe since it's no longer shared.
        lora.scale = base_lora_scale(lora.alpha, lora.rank) * strength_start
        new_transformer = ASDX_LoraLoader._apply_lora_to_transformer(
            model["transformer"], lora, model["config"]
        )
        bridge.clear_mlx_cache()
        new_model = {**model, "transformer": new_transformer}

        # Store schedule info on the model — the sampler reads this and
        # adjusts LoRA strength per step (see sampler/core.py).
        new_model["lora_schedule"] = {
            "name": lora_name,
            "lora": lora,
            "strength_start": strength_start,
            "strength_end": strength_end,
            "strength_middle": strength_middle,
            "strength_curve": strength_curve,
        }

        print(f"[ASDX] LoRA schedule: '{lora_name}' "
              f"start={strength_start:.2f} middle={strength_middle:.2f} end={strength_end:.2f}")

        return (new_model,)

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
