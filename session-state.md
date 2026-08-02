---
name: session-state-advanced-features
description: Session state after integrating advanced features (LoRA, ControlNet, IP-Adapter, TeaCache, Kontext)
metadata:
  type: reference
  date: 2026-08-02
---

## Session Summary

**Date:** 2026-08-02
**Goal:** Integrate advanced features from SDMLX into ASDX to make it complete.

### What was done
- Integrated LoRA runtime loading (standard A/B + ComfyUI diff format)
- Integrated TeaCache acceleration (output-level step skipping)
- Integrated Kontext KV cache (reference image conditioning)
- Integrated ControlNet Union (8 control types)
- Integrated IP-Adapter cross-attention injection
- Updated native transformer to pass kontext_kv through attention layers
- Updated README.md with all new features
- Created git repo and committed

### Current state
- All 14 Python files pass py_compile verification
- Git repo: 2 commits on `master`
- All tasks completed (tasks #2-#9)

### Key decisions
- TeaCache uses accumulated L1 norm with adaptive threshold interpolation
- Kontext stores K/V per transformer layer with configurable token count
- LoRA supports both standard LoRA (A/B matrices) and ComfyUI diff format
- ControlNet Union maps 8 control types (pose, depth, soft_edge, line_canny, normal, segment, tile, repaint)
- IP-Adapter uses CLIP-Vision-H encoder with projection weights

### Pending / Future work
- FLUX.2 / Klein model support (partial - kontext support added, full model not yet implemented)
- Web UI for ControlNet/IP-Adapter parameters
- Performance benchmarking
