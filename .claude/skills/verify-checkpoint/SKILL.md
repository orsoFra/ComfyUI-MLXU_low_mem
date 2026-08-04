---
name: verify-checkpoint
description: Run this project's established 4-step checkpoint verification recipe (py_compile, random-weight forward pass, real-checkpoint matched N/M load, std-vs-random-init sanity check) against a native/<x>/ model family. Use after implementing or modifying a native MLX architecture/weight-map, before claiming a family "works" against real weights.
---

# verify-checkpoint

This project's non-negotiable verification recipe for any `native/<x>/` model family,
established in Session 11 after a checkpoint-key comparison bug (comparing tuples instead
of strings) silently matched zero weights while reporting no error — the model ran forward
on 100% randomly-initialized weights and nothing in the pipeline raised. Every family ported
since (Krea2, SDXL, FLUX.1, Z-Image) has been verified with all 4 steps below before being
considered done.

## The 4 checks, in order

1. **`python3 -m py_compile`** on every file in `native/<x>/` — syntax guard, catches nothing
   deep but is free and first.
2. **Forward pass on a reduced random-weight config** — fewer blocks/layers than the real
   model, random init, just checking shapes flow through and the output is NaN/Inf-free.
   Watch for hidden minimum-size requirements before assuming a NaN means a real bug: this
   project has hit two false alarms from configs that were "too reduced" (SDXL's
   `transformer_depth_output` derivation, Z-Image's `adaLN_modulation` needing the timestep
   embedder's hardcoded 256-dim output regardless of the reduced `dim`).
3. **Real checkpoint load with `matched N/M` logging** — every family's `load_<x>_transformer()`
   must log how many of the model's flattened parameter keys were found in the checkpoint's
   state dict. Anything less than 100% matched on a real, correctly-identified checkpoint is a
   bug to chase down before proceeding, not a warning to note and move past.
4. **std (loaded) vs std (random init) sanity check** — for ~5-7 weights spanning different
   depths, compare the loaded tensor's std against the characteristic std of MLX's default
   `nn.Linear` init, `1/sqrt(3*fan_in)`. This is the check that would have caught the Session
   11 bug even if step 3's match count had been (wrongly) computed as 100% — it is the only
   step that looks at the actual numbers, not just whether a key existed.

   **Do not over-trigger on this check.** A SINGLE input-projection layer
   (x_embedder/t_embedder-style) landing close to its random-init std is a known, benign
   phenomenon seen in real trained checkpoints (confirmed for Z-Image, Session 14) — it is
   not evidence of a loading bug by itself. Only treat it as a failure if MULTIPLE
   independent, deeper layers (attention projections, MLP layers, final output layer) are
   all suspiciously close to random-init std.

## How to run it

1. Copy `scripts/verify_checkpoint_template.py` to the repo root (or the scratchpad) as
   `verify_<family>.py`.
2. Fill in the 3 `TODO` sections: import paths/class names at the top, a reduced config +
   minimal forward-pass call matching the family's actual `predict()`/`__call__` signature,
   and 5-7 `layer_paths` for the std check spanning different network depths.
3. Run it against a real checkpoint on this machine:
   ```
   python3 verify_<family>.py --checkpoint /path/to/model.safetensors --dtype float16
   ```
4. If no real checkpoint exists for the family on this machine, say so explicitly rather than
   skipping the check silently — this project has previously documented such gaps rather than
   pretending full verification happened (e.g. FLUX.2/Klein has no real checkpoint on this
   machine as of the last multi-model planning session).
5. Delete the copied script once verification is recorded (in the session-state memory or the
   PR description) — do not let it join `test1.py`..`test4.py` as permanent root clutter.

## Full end-to-end verification (beyond this recipe)

These 4 checks validate the architecture and weight-loading in isolation. They do NOT
constitute a real txt2img test — that requires a live ComfyUI instance with real CLIP/VAE,
which has been an explicitly open item for every family ported so far in this project. Do not
claim a family "works" or is "done" based on the 4 checks alone; report them as structural
verification with the visual end-to-end test still open, unless that test was actually run.
