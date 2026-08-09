"""
Memory Profiler & Cache Manager
================================
Nodes for monitoring and managing Apple Silicon Unified Memory.

Provides real-time memory statistics and cache control for the MLX
runtime, essential for debugging OOM conditions on devices with
limited RAM (8GB/16GB MacBook Air, etc.).
"""

from __future__ import annotations

import gc
import time
from typing import Any

from comfy_api.latest import io

import mlx.core as mx
import torch


# ── Memory Profiler ───────────────────────────────────────────────────

class ASDX_MemoryProfiler(io.ComfyNode):
    """Profile and display Apple Silicon Unified Memory usage.

    Reports:
      - MLX active memory (in use by computations)
      - MLX cache memory (allocated but idle)
      - MLX peak memory (highest water mark)
      - PyTorch MPS cached/allocated memory
      - System-level memory pressure estimate

    Output is returned as a string slot for logging/debugging.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_MemoryProfiler",
            display_name="🍏 ASDX Memory Profiler",
            category="ASDX/Utilities",
            inputs=[
                io.Combo.Input(
                    "mode", options=["snapshot", "alloc", "free"], default="snapshot",
                    tooltip="snapshot: read stats, alloc: trigger allocation, free: clear caches",
                ),
                io.Int.Input("alloc_mb", default=256, min=0, max=8192, step=64),
            ],
            outputs=[
                io.String.Output(display_name="report"),
                io.Float.Output(display_name="active_gb"),
                io.Float.Output(display_name="cache_gb"),
                io.Float.Output(display_name="peak_gb"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, mode: str, alloc_mb: int) -> Any:
        # A memory snapshot must be taken fresh on every run -- if ComfyUI
        # reused a cached result because mode/alloc_mb didn't change, the
        # reported stats would silently go stale instead of reflecting the
        # current session's actual memory state.
        return time.time()

    @classmethod
    def execute(cls, mode: str, alloc_mb: int) -> io.NodeOutput:
        if mode == "alloc":
            # Allocate some memory to trigger reporting
            dummy = mx.zeros((alloc_mb * 1024 * 1024 // 4,))  # int32
            mx.eval(dummy)
            del dummy

        stats = cls._collect_stats()

        report_lines = [
            f"=== ASDX Memory Report [{mode}] ===",
            f"MLX Active:  {stats['mlx_active_gb']:.2f} GB",
            f"MLX Cache:   {stats['mlx_cache_gb']:.2f} GB",
            f"MLX Peak:    {stats['mlx_peak_gb']:.2f} GB",
            f"MLX Limit:   {stats['mlx_limit_gb']:.2f} GB",
        ]

        if torch.backends.mps.is_available():
            stats_torch = cls._torch_mps_stats()
            report_lines.extend([
                f"MPS Allocated: {stats_torch['allocated_gb']:.2f} GB",
                f"MPS Cached:    {stats_torch['cached_gb']:.2f} GB",
            ])

        # Estimate total used
        total_used = (
            stats['mlx_active_gb'] +
            stats['mlx_cache_gb'] +
            (stats_torch['allocated_gb'] if torch.backends.mps.is_available() else 0)
        )
        report_lines.append(f"Total Estimated: {total_used:.2f} GB")
        report_lines.append(f"Timestamp: {time.strftime('%H:%M:%S')}")

        report = "\n".join(report_lines)
        print(report)

        return io.NodeOutput(
            report,
            stats['mlx_active_gb'],
            stats['mlx_cache_gb'],
            stats['mlx_peak_gb'],
        )

    @staticmethod
    def _collect_stats() -> dict[str, float]:
        """Collect MLX memory statistics."""
        active = mx.get_active_memory() / (1024 ** 3)
        cache = mx.get_cache_memory() / (1024 ** 3)
        peak = mx.get_peak_memory() / (1024 ** 3)
        limit = mx.constant_cache_limit() / (1024 ** 3) if hasattr(mx, 'constant_cache_limit') else 0.0

        return {
            "mlx_active_gb": round(active, 2),
            "mlx_cache_gb": round(cache, 2),
            "mlx_peak_gb": round(peak, 2),
            "mlx_limit_gb": round(limit, 2),
        }

    @staticmethod
    def _torch_mps_stats() -> dict[str, float]:
        """Collect PyTorch MPS memory statistics."""
        try:
            allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
            cached = torch.mps.current_reserved_memory() / (1024 ** 3)
            return {
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
            }
        except Exception:
            return {"allocated_gb": 0.0, "cached_gb": 0.0}


# ── Cache Clearer (utility) ──────────────────────────────────────────

class ASDX_CacheManager(io.ComfyNode):
    """Clear MLX and PyTorch MPS caches.

    Use between major workflow phases (e.g., after loading a model,
    before starting sampling) to minimize memory pressure.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ASDX_CacheManager",
            display_name="🍏 ASDX Cache Manager",
            category="ASDX/Utilities",
            inputs=[
                io.Boolean.Input("clear_mlx", default=True),
                io.Boolean.Input("clear_mps", default=True),
                io.Boolean.Input("gc_collect", default=True),
            ],
            outputs=[
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, clear_mlx: bool, clear_mps: bool, gc_collect: bool) -> Any:
        # Cache-clearing is a side effect that must run every time this node
        # executes, not just the first time a given combination of booleans
        # is queued -- see ASDX_MemoryProfiler.fingerprint_inputs above.
        return time.time()

    @classmethod
    def execute(cls, clear_mlx: bool, clear_mps: bool, gc_collect: bool) -> io.NodeOutput:
        freed = []

        if clear_mlx:
            before = mx.get_cache_memory()
            mx.clear_cache()
            freed.append(f"MLX cache cleared ({(before - mx.get_cache_memory()) / 1024**3:.1f}GB)")

        if clear_mps and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                freed.append("MPS cache cleared")
            except Exception as e:
                freed.append(f"MPS cache clear: {e}")

        if gc_collect:
            before_gc = gc.collect()
            freed.append(f"Python GC: {before_gc} objects collected")

        report = f"[ASDX] Cache cleared: {'; '.join(freed)}"
        print(report)

        return io.NodeOutput(report)


NODE_LIST = [
    ASDX_MemoryProfiler,
    ASDX_CacheManager,
]
