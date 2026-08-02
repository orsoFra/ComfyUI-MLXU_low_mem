"""Caching mechanisms for the MLX sampler.

Contains:
  - TeaCacheState: output-level step skipping via accumulated L1 norm
  - KontextCache: KV cache for reference image conditioning
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx


# ── TeaCache State ────────────────────────────────────────────────────

class TeaCacheState:
    """TeaCache state for output-level step skipping.

    TeaCache works by comparing the output of consecutive transformer steps.
    When the accumulated L1 difference stays below a threshold, the step
    is skipped and the previous output is reused.
    """

    def __init__(
        self,
        threshold: float = 0.08,
        threshold_end: float | None = None,
        warmup_steps: int = 5,
        final_steps: int = 3,
    ):
        self.threshold = threshold
        self.threshold_end = threshold_end or threshold * 2.0
        self.warmup_steps = warmup_steps
        self.final_steps = final_steps
        self.accumulated: float = 0.0
        self.previous_probe: mx.array | None = None
        self.last_feature: mx.array | None = None
        self.hits: int = 0
        self.real_steps: int = 0
        self.total_steps: int = 0
        self.metrics: list[dict[str, Any]] = []

    def step_threshold(self, step: int) -> float:
        """Compute adaptive threshold that interpolates between start and end."""
        if self.total_steps <= 0:
            return self.threshold
        active_start = max(1, self.warmup_steps + 1)
        active_end = max(active_start, self.total_steps - self.final_steps)
        if active_end <= active_start:
            return self.threshold
        progress = (step - active_start) / (active_end - active_start)
        progress = max(0.0, min(1.0, progress))
        return self.threshold + (self.threshold_end - self.threshold) * progress

    def try_reuse(
        self,
        current_output: mx.array,
        step: int,
        total_steps: int,
    ) -> tuple[mx.array, bool, str]:
        """Try to reuse previous output. Returns (output, skipped, reason)."""
        self.total_steps = total_steps

        # Warmup: always compute
        if step <= self.warmup_steps:
            self._record_real(current_output, step, "warmup")
            return current_output, False, "warmup"

        # Final steps: always compute for quality
        if total_steps > 0 and step > total_steps - self.final_steps:
            self._record_real(current_output, step, "final_guard")
            return current_output, False, "final_guard"

        # First step: no previous to compare
        if self.previous_probe is None:
            self._record_real(current_output, step, "first_probe")
            return current_output, False, "first_probe"

        # Compute L1 difference
        delta = float(mx.mean(mx.abs(current_output - self.previous_probe)).item())
        rel_l1 = delta / (float(mx.mean(mx.abs(self.previous_probe)).item()) + 1e-6)

        accumulated_next = self.accumulated + rel_l1
        threshold = self.step_threshold(step)

        if accumulated_next <= threshold:
            # Skip: reuse previous output
            self.accumulated = accumulated_next
            self.hits += 1
            self.metrics.append({
                "step": step, "action": "reuse", "rel_l1": rel_l1,
                "accumulated": accumulated_next, "threshold": threshold,
            })
            return self.last_feature, True, f"cache(rel_l1={rel_l1:.4f})"

        # Threshold exceeded: compute real, reset accumulator
        self._record_real(current_output, step, "threshold")
        return current_output, False, f"real(rel_l1={rel_l1:.4f})"

    def _record_real(self, feature: mx.array, step: int, reason: str) -> None:
        """Record a real (non-skipped) step's output."""
        self.previous_probe = feature
        self.last_feature = feature
        self.accumulated = 0.0
        self.real_steps += 1
        self.metrics.append({
            "step": step, "action": "real", "reason": reason,
        })


# ── Kontext KV Cache ──────────────────────────────────────────────────

class KontextCache:
    """KV cache for reference image tokens in transformer attention.

    Caches K/V pairs from a reference image's latent encoding, then
    appends them to the attention computation at each denoising step.
    """

    def __init__(self):
        self.enabled: bool = False
        self.cache: dict[str, tuple[mx.array, mx.array]] = {}  # layer_idx -> (k, v)
        self.reference_tokens: int = 0
        self.hits: int = 0
        self.stores: int = 0

    def reset(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.stores = 0

    def set(self, enabled: bool, reference_tokens: int) -> None:
        self.enabled = bool(enabled and reference_tokens > 0)
        self.reset()
        if self.enabled:
            self.reference_tokens = reference_tokens

    def ready(self) -> bool:
        return self.enabled and bool(self.cache)

    def get_attention(
        self,
        layer_idx: int,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        """Get attention output, using cached reference K/V if available."""
        if not self.enabled:
            return self._plain_attention(q, k, v)

        cache_key = str(layer_idx)
        cached = self.cache.get(cache_key)

        if cached is not None:
            # Cache hit: append cached K/V
            self.hits += 1
            k_full = mx.concatenate([k, cached[0]], axis=2)
            v_full = mx.concatenate([v, cached[1]], axis=2)
            return self._plain_attention(q, k_full, v_full)

        # Cache miss: store reference K/V if we have enough tokens
        if k.shape[2] >= self.reference_tokens:
            ref_k = mx.contiguous(k[:, :, -self.reference_tokens:, :])
            ref_v = mx.contiguous(v[:, :, -self.reference_tokens:, :])
            mx.eval(ref_k, ref_v)
            self.cache[cache_key] = (ref_k, ref_v)
            self.stores += 1

        return self._plain_attention(q, k, v)

    @staticmethod
    def _plain_attention(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """Standard scaled dot-product attention."""
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        attn = mx.softmax((q * scale) @ k.transpose(0, 1, 3, 2), axis=-1)
        return attn @ v
