"""Live preview registry for step-by-step sampling feedback.

Mirrors the live preview callback pattern from ComfyUI-mflux-AnyModel:
a registry that fires callbacks after each denoising step, enabling
real-time progress visualization and early cancellation.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import torch
from comfy_api.latest import io


# ── LivePreviewRegistry ───────────────────────────────────────────────

class LivePreviewRegistry:
    """Registry for step-by-step preview callbacks.

    After each denoising step, the sampler can call fire() to notify
    all registered callbacks with the current state.
    """

    _callbacks: list[Callable] = []

    @classmethod
    def register(cls, callback: Callable) -> None:
        """Register a callback to be called on each step."""
        if callback not in cls._callbacks:
            cls._callbacks.append(callback)

    @classmethod
    def unregister(cls, callback: Callable) -> None:
        """Remove a previously registered callback."""
        cls._callbacks.remove(callback)

    @classmethod
    def fire(
        cls,
        step: int,
        total_steps: int,
        latent: mx.array,
        height: int,
        width: int,
    ) -> None:
        """Fire all registered callbacks with current sampling state."""
        for cb in cls._callbacks:
            try:
                cb(step, total_steps, latent, height, width)
            except Exception:
                pass  # Don't let one bad callback break sampling

    @classmethod
    def clear(cls) -> None:
        """Remove all registered callbacks."""
        cls._callbacks.clear()

    @classmethod
    def count(cls) -> int:
        return len(cls._callbacks)


# ── Node: LivePreview (debug/inspection) ──────────────────────────────

class ASDX_LivePreview(io.ComfyNode):
    """Register a live preview callback for the current sampling run.

    This node is primarily for debugging and advanced workflow integration.
    It registers a callback that fires after each denoising step.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_LivePreview",
            display_name="🍏 ASDX Live Preview",
            category="ASDX/Utilities",
            inputs=[
                io.Combo.Input("callback_type", options=["log", "progress", "none"], default="log"),
            ],
            outputs=[
                io.Custom("ASDX_PREVIEW_HANDLE").Output(display_name="handle"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, callback_type: str) -> Any:
        # Registration is a side effect (LivePreviewRegistry.clear() +
        # register()) that must happen every run, not just the first time a
        # given callback_type is queued -- same reasoning as
        # ASDX_MemoryProfiler.fingerprint_inputs (memory.py).
        import time
        return time.time()

    @classmethod
    def execute(cls, callback_type: str) -> io.NodeOutput:
        """Register a preview callback."""
        # Each run creates a brand-new closure, so the registry's own
        # dedup check (`if callback not in cls._callbacks`) never catches
        # it -- without this, every generation using this node permanently
        # appends another callback that keeps firing (and doing real work,
        # for "progress") on every future run's sampling steps. Only one
        # active callback for this node is the intended behavior.
        LivePreviewRegistry.clear()

        handle: dict[str, Any] = {"type": callback_type, "steps": 0}

        if callback_type == "log":
            def _cb(step: int, total: int, latent: mx.array, h: int, w: int):
                print(f"[ASDX Preview] Step {step + 1}/{total} "
                      f"({h}x{w})")
                handle["steps"] = step + 1

            LivePreviewRegistry.register(_cb)
        elif callback_type == "progress":
            import comfy.utils

            def _cb(step: int, total: int, latent: mx.array, h: int, w: int):
                try:
                    # Decode latent to preview bytes
                    from . import bridge
                    samples = bridge._unpack_flux_latents(latent, h, w)
                    import comfy.latent_formats
                    samples = comfy.latent_formats.Flux().process_out(samples)
                    if torch.backends.mps.is_available():
                        samples = samples.to(device="mps")
                    # Encode to JPEG for progress bar
                    import io
                    from PIL import Image
                    img_np = samples[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=85)
                    preview_bytes = buf.getvalue()
                    comfy.utils.ProgressBar(total).update_absolute(
                        step + 1, total, preview_bytes
                    )
                except Exception:
                    pass
                handle["steps"] = step + 1

            LivePreviewRegistry.register(_cb)

        return io.NodeOutput(handle)


# ── Node Mappings ─────────────────────────────────────────────────────

NODE_LIST = [
    ASDX_LivePreview,
]
