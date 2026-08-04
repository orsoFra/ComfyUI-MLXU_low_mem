"""Caching mechanisms for the MLX sampler.

Contains:
  - TeaCacheState: output-level step skipping via accumulated L1 norm
"""

from __future__ import annotations

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
