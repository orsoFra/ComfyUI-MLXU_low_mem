#!/usr/bin/env python3
"""Promote staged memory observations into the versioned evidence file.

Run this deliberately after collecting real observations (via
`ASDX_RECORD_MEMORY_EVIDENCE=1` during test loads) -- it never runs
automatically. A key is promoted to "measured" only when it has at least
MIN_CONSISTENT_RUNS observations whose max/min stay within
CONSISTENCY_TOLERANCE of the median; a single outlier run does not fix the
recorded value. The promoted value is the MAXIMUM observed peak among the
accepted runs (this gates against OOM, not an average-case report).

Usage:
    python3 scripts/promote_memory_evidence.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent.parent))

from apple_silicon_nodes.memory_calibration import (  # noqa: E402
    CONSISTENCY_TOLERANCE,
    MIN_CONSISTENT_RUNS,
    _EVIDENCE_PATH,
    _STAGING_PATH,
    _load_evidence_store,
    _save_evidence_store,
)


def _load_staged_runs() -> dict[str, list[dict]]:
    by_key: dict[str, list[dict]] = {}
    if not _STAGING_PATH.exists():
        return by_key
    with open(_STAGING_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_key.setdefault(record["key"], []).append(record["run"])
    return by_key


def _is_consistent(peaks: list[int]) -> bool:
    if not peaks:
        return False
    m = median(peaks)
    if m == 0:
        return False
    return (max(peaks) - min(peaks)) / m <= CONSISTENCY_TOLERANCE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    staged = _load_staged_runs()
    if not staged:
        print(f"No staged observations found at {_STAGING_PATH}")
        return 0

    store = _load_evidence_store()
    entries = store.setdefault("entries", {})
    promoted, skipped = [], []

    for key, runs in staged.items():
        peaks = [r["peak_bytes"] for r in runs]
        recent = runs[-MIN_CONSISTENT_RUNS:]
        recent_peaks = [r["peak_bytes"] for r in recent]

        if len(runs) < MIN_CONSISTENT_RUNS:
            skipped.append((key, f"only {len(runs)}/{MIN_CONSISTENT_RUNS} runs"))
            continue
        if not _is_consistent(recent_peaks):
            spread = (max(recent_peaks) - min(recent_peaks)) / median(recent_peaks)
            skipped.append((key, f"inconsistent: {spread*100:.0f}% spread over "
                                  f"last {len(recent)} runs (tolerance {CONSISTENCY_TOLERANCE*100:.0f}%)"))
            continue

        promoted_peak = max(recent_peaks)
        entries[key] = {
            "status": "measured",
            "sample_count": len(recent),
            "peak_bytes": promoted_peak,
            "runs": recent,
        }
        promoted.append((key, promoted_peak / (1024 ** 3), len(recent)))

    for key, peak_gb, n in promoted:
        print(f"PROMOTED  {key}: {peak_gb:.1f}GB (n={n})")
    for key, reason in skipped:
        print(f"SKIPPED   {key}: {reason}")

    if not promoted:
        print("Nothing promoted.")
        return 0

    if args.dry_run:
        print(f"\n--dry-run: not writing to {_EVIDENCE_PATH}")
        return 0

    _save_evidence_store(store)
    print(f"\nWrote {len(promoted)} promoted entrie(s) to {_EVIDENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
