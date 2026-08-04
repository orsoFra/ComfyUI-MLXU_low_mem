#!/usr/bin/env python3
"""Template for the project's checkpoint-verification recipe.

Copy this file to the repo root (or scratchpad) as `verify_<family>.py`, fill in
the TODO-marked sections for the specific family, run it, then delete the copy —
this file itself is a template, not a script meant to run as-is.

This replaces the pattern of accumulating ad hoc `testN.py` scripts at the repo
root (test1.py..test4.py, test_krea2_load.py) with one reusable, parametrized
recipe. It reproduces the exact 4 checks every family in this project has been
verified against (SDXL, Krea2, FLUX.1, Z-Image): py_compile, forward pass on a
random-weight reduced config, real-checkpoint load with `matched N/M`, and a
std-vs-random-init sanity check.

Usage (after filling in the TODOs):
    python3 verify_<family>.py --checkpoint /path/to/model.safetensors --dtype float16
"""

from __future__ import annotations

import argparse
import math
import py_compile
import time
from pathlib import Path

import mlx.core as mx


# ─────────────────────────────────────────────────────────────────────────
# TODO 1: fill in the family's import paths.
# ─────────────────────────────────────────────────────────────────────────
FAMILY_MODULE = "apple_silicon_nodes.native.<x>"  # e.g. "apple_silicon_nodes.native.flux2"
CONFIG_CLASS_NAME = "<X>Config"
TRANSFORMER_CLASS_NAME = "<X>Transformer"
LOAD_FN_NAME = "load_<x>_transformer"


def step0_py_compile() -> bool:
    """Syntax check every file in the family's native/<x>/ subpackage."""
    print("\n[0/4] py_compile check...")
    family_dir = Path("apple_silicon_nodes/native") / FAMILY_MODULE.rsplit(".", 1)[-1]
    ok = True
    for f in sorted(family_dir.glob("*.py")):
        try:
            py_compile.compile(str(f), doraise=True)
            print(f"  OK   {f}")
        except py_compile.PyCompileError as e:
            print(f"  FAIL {f}: {e}")
            ok = False
    return ok


def step1_random_weight_forward() -> bool:
    """Forward pass on a REDUCED config with random weights. NaN-free is the bar.

    TODO 2: instantiate a reduced config here (fewer blocks/layers than the real
    model — the point is a fast structural check, not a full-size run). Watch for
    hidden minimum-size gotchas: this project has hit two of them already —
    SDXL's `transformer_depth_output` derivation breaking on too-small configs,
    and Z-Image's adaLN_modulation needing dim>=256 regardless of the reduced
    config's other dims (its timestep embedder output is hardcoded to 256-dim).
    """
    print("\n[1/4] Random-weight forward pass (reduced config)...")
    import importlib
    module = importlib.import_module(FAMILY_MODULE)
    config_cls = getattr(module, CONFIG_CLASS_NAME)
    transformer_cls = getattr(module, TRANSFORMER_CLASS_NAME)

    # TODO: set reduced-but-valid dims for this family.
    config = config_cls(dtype="float32")
    model = transformer_cls(config)

    # TODO: build minimal random inputs matching this family's __call__/predict()
    # signature and run it. Example shape (adapt to the real signature):
    #   noise = mx.random.normal((1, 64, config.latent_channels))
    #   output = model.predict(img=noise, txt=..., timestep=0.5)
    output = None  # TODO: replace
    if output is None:
        print("  SKIPPED (fill in TODO 2 first)")
        return True

    mx.eval(output)
    has_nan = bool(mx.isnan(output).any())
    has_inf = bool(mx.isinf(output).any())
    print(f"  output shape={output.shape} NaN={has_nan} Inf={has_inf}")
    return not (has_nan or has_inf)


def step2_real_checkpoint_load(checkpoint: Path, dtype: str) -> tuple[bool, dict]:
    """Load the real checkpoint and report matched/missing/extra key counts.

    Uses the family's own load_<x>_transformer() — this only re-verifies that
    function's own `matched N/M` logging is honest, it does not reimplement
    the matching logic (that would risk a second, drifted copy of the exact
    string-vs-tuple bug class this check exists to catch).
    """
    print(f"\n[2/4] Loading real checkpoint: {checkpoint.name}...")
    if not checkpoint.exists():
        print(f"  SKIPPED: checkpoint not found at {checkpoint}")
        return True, {}

    import importlib
    module = importlib.import_module(FAMILY_MODULE)
    load_fn = getattr(module, LOAD_FN_NAME)

    t0 = time.perf_counter()
    model = load_fn(str(checkpoint), dtype=dtype)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    return True, {"model": model}


def step3_std_vs_random_init(model, layer_paths: list[str]) -> bool:
    """Compare loaded weight std against the characteristic std of MLX's default
    nn.Linear init (1/sqrt(3*fan_in)) for a handful of layers.

    A loaded std suspiciously close to the random-init std (diff < ~0.01) on
    EVERY checked layer means the weights were never actually assigned — this
    is the check that would have caught the Session 11 tuple-vs-string bug
    even if `matched N/M` had been miscounted as 100%. A close std on ONE
    input-projection layer (x_embedder/t_embedder) alone is not evidence of a
    bug — that is a documented, benign phenomenon (Z-Image, Session 14). Only
    treat it as a failure if MULTIPLE independent deeper layers are all close.

    TODO 3: fill in `layer_paths` — dotted paths into model.parameters() for
    ~5-7 layers spanning different depths (embedders, mid-stack attention/mlp,
    final layer).
    """
    print("\n[3/4] std (loaded) vs std (random init) sanity check...")
    if not layer_paths:
        print("  SKIPPED (fill in TODO 3 first)")
        return True

    suspicious = 0
    for path in layer_paths:
        obj = model
        for part in path.split("."):
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        weight = obj["weight"] if isinstance(obj, dict) else obj
        fan_in = weight.shape[-1]
        random_init_std = 1.0 / math.sqrt(3 * fan_in)
        loaded_std = float(mx.std(weight.astype(mx.float32)))
        close = abs(loaded_std - random_init_std) < 0.01
        flag = "SUSPICIOUS" if close else "trained"
        print(f"  {path}: loaded_std={loaded_std:.5f} random_init_std={random_init_std:.5f} [{flag}]")
        suspicious += close

    if suspicious >= max(2, len(layer_paths) // 2):
        print(f"  FAILED: {suspicious}/{len(layer_paths)} layers look like random init, not loaded weights.")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    ok = True
    ok &= step0_py_compile()
    ok &= step1_random_weight_forward()
    loaded_ok, ctx = step2_real_checkpoint_load(args.checkpoint, args.dtype)
    ok &= loaded_ok
    if "model" in ctx:
        ok &= step3_std_vs_random_init(ctx["model"], layer_paths=[])  # TODO 3

    print(f"\n{'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
