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

import mlx.core as mx
import torch


# ── Memory Profiler ───────────────────────────────────────────────────

class ASDX_MemoryProfiler:
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["snapshot", "alloc", "free"], {
                    "default": "snapshot",
                    "description": "snapshot: read stats, alloc: trigger allocation, free: clear caches"
                }),
                "alloc_mb": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("report", "active_gb", "cache_gb", "peak_gb")
    FUNCTION = "profile"
    CATEGORY = "ASDX/Utilities"

    def profile(self, mode: str, alloc_mb: int) -> tuple[str, float, float, float]:
        if mode == "alloc":
            # Allocate some memory to trigger reporting
            dummy = mx.zeros((alloc_mb * 1024 * 1024 // 4,))  # int32
            mx.eval(dummy)
            del dummy

        stats = self._collect_stats()

        report_lines = [
            f"=== ASDX Memory Report [{mode}] ===",
            f"MLX Active:  {stats['mlx_active_gb']:.2f} GB",
            f"MLX Cache:   {stats['mlx_cache_gb']:.2f} GB",
            f"MLX Peak:    {stats['mlx_peak_gb']:.2f} GB",
            f"MLX Limit:   {stats['mlx_limit_gb']:.2f} GB",
        ]

        if torch.backends.mps.is_available():
            stats_torch = self._torch_mps_stats()
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

        return (
            report,
            stats['mlx_active_gb'],
            stats['mlx_cache_gb'],
            stats['mlx_peak_gb'],
        )

    def _collect_stats(self) -> dict[str, float]:
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

class ASDX_CacheManager:
    """Clear MLX and PyTorch MPS caches.

    Use between major workflow phases (e.g., after loading a model,
    before starting sampling) to minimize memory pressure.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_mlx": ("BOOLEAN", {"default": True}),
                "clear_mps": ("BOOLEAN", {"default": True}),
                "gc_collect": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "clear"
    CATEGORY = "ASDX/Utilities"

    def clear(self, clear_mlx: bool, clear_mps: bool, gc_collect: bool) -> tuple[str]:
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

        return (report,)


NODE_CLASS_MAPPINGS = {
    "ASDX_MemoryProfiler": ASDX_MemoryProfiler,
    "ASDX_CacheManager": ASDX_CacheManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASDX_MemoryProfiler": "🍏 ASDX Memory Profiler",
    "ASDX_CacheManager": "🍏 ASDX Cache Manager",
}
