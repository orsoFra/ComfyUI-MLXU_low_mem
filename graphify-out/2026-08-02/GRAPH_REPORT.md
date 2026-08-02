# Graph Report - .  (2026-08-02)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 435 nodes · 661 edges · 20 communities (18 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `eb4192a2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- controlnet.py
- ASDX - Apple Silicon Diffusion Nodes
- .sample
- ip_adapter.py
- lora.py
- apple_silicon_nodes/__init__.py
- conditioning.py
- bridge.py
- mlx_vae.py
- vae.py
- array
- .__init__
- FluxConfig
- ASDX_CheckpointLoader
- native/__init__.py
- FluxTransformer
- ASDX_ApplyControlNet
- apply_rope
- .mlx_dtype
- Any

## God Nodes (most connected - your core abstractions)
1. `ASDX - Apple Silicon Diffusion Nodes` - 31 edges
2. `FluxConfig` - 20 edges
3. `FluxTransformer` - 16 edges
4. `ControlNetUnionModel` - 11 edges
5. `load_transformer()` - 11 edges
6. `KontextCache` - 10 edges
7. `ASDX_MLXSampler` - 10 edges
8. `ASDX_LoraLoader` - 8 edges
9. `ResBlock` - 8 edges
10. `_assign_weights()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `ASDX - Apple Silicon Diffusion Nodes` --follows--> `ComfyUI node conventions (INPUT_TYPES, RETURN_TYPES, etc.)`  [EXTRACTED]
  README.md → CLAUDE.md
- `ASDX - Apple Silicon Diffusion Nodes` --pending_support--> `FLUX.2 / Klein model support (pending)`  [EXTRACTED]
  README.md → session-state.md
- `ASDX - Apple Silicon Diffusion Nodes` --supports--> `LoRA formats (A/B + ComfyUI diff)`  [EXTRACTED]
  README.md → session-state.md
- `ASDX - Apple Silicon Diffusion Nodes` --uses--> `Tensor format [B,H,W,C] -> [B,C,H,W]`  [EXTRACTED]
  README.md → CLAUDE.md
- `MLX framework` --bridges_to--> `PyTorch-MLX bridge`  [EXTRACTED]
  README.md → CLAUDE.md

## Import Cycles
- None detected.

## Communities (20 total, 2 thin omitted)

### Community 0 - "controlnet.py"
Cohesion: 0.07
Nodes (29): ASDX_ControlNetUnionLoader, _assign_controlnet_weights(), ControlNetCondEmbedding, ControlNetUnionModel, load_controlnet_union(), Any, array, Path (+21 more)

### Community 1 - "ASDX - Apple Silicon Diffusion Nodes"
Cohesion: 0.07
Nodes (34): Apple Silicon (M1-M5), ASDX - Apple Silicon Diffusion Nodes, ASDX node catalog, PyTorch-MLX bridge, Canon protocol, CLIP (text + vision), ComfyUI, ComfyUI node conventions (INPUT_TYPES, RETURN_TYPES, etc.) (+26 more)

### Community 2 - ".sample"
Cohesion: 0.08
Nodes (21): ASDX_MLXSampler, KontextCache, array, Dtype, Tensor, Record a real (non-skipped) step's output., KV cache for reference image tokens in transformer attention.      Caches K/V pa, Get attention output, using cached reference K/V if available. (+13 more)

### Community 3 - "ip_adapter.py"
Cohesion: 0.07
Nodes (22): ASDX_ApplyIPAdapter, ASDX_IPAdapterCLIPVisionEncode, ASDX_IPAdapterLoader, CLIPVisionEncoder, IPAdapterCache, Any, array, Path (+14 more)

### Community 4 - "lora.py"
Cohesion: 0.08
Nodes (22): ASDX_LoraLoader, ASDX_LoraSchedule, ASDX_MultiLoraLoader, LoRAAdapter, LoRATarget, Any, Path, LoRA Runtime Loading ==================== Load and apply LoRA adapters to FLUX t (+14 more)

### Community 5 - "apple_silicon_nodes/__init__.py"
Cohesion: 0.08
Nodes (14): Apple Silicon Diffusion Nodes ============================= Custom ComfyUI nodes, _Unavailable, ASDX_EmptyFLUXLatent, Empty FLUX Latent ================= Creates empty 16-channel FLUX latents optimi, Create an empty 16-channel FLUX latent tensor.      The latent is placed on the, Get the best available device., ASDX_CacheManager, ASDX_MemoryProfiler (+6 more)

### Community 6 - "conditioning.py"
Cohesion: 0.10
Nodes (13): Any, ASDX_CLIPLoader, ASDX_CLIPTextEncode, ASDX_ConditioningMerger, ASDX_DualCLIPLoader, _clip_type_from_string(), Conditioning nodes ================== CLIP text encoding and conditioning manipu, Encode text prompts to conditioning for any model type.      Auto-detects FLUX v (+5 more)

### Community 7 - "bridge.py"
Cohesion: 0.11
Nodes (26): clear_mlx_cache(), collect_mlx_memory(), conditioning_to_mlx(), mlx_to_comfy_image(), mlx_to_comfy_latent(), prepare_noise_from_latent(), Any, array (+18 more)

### Community 8 - "mlx_vae.py"
Cohesion: 0.13
Nodes (17): get_vae_decoder(), GroupNorm, _make_down_block(), _make_up_block(), array, MLX VAE Decoder/Encoder ======================== Lightweight FLUX VAE implementa, Get the cached VAE decoder instance., Clear cached VAE instances. (+9 more)

### Community 9 - "vae.py"
Cohesion: 0.15
Nodes (13): get_vae_encoder(), Get the cached VAE encoder instance., ASDX_VAEDecode, ASDX_VAEEncode, Any, array, Tensor, VAE Decode / Encode nodes ========================= MLX-native VAE decoding and (+5 more)

### Community 10 - "array"
Cohesion: 0.13
Nodes (11): array, Returns a tuple of (param_i * x) for each of num_params., Args:             img: [B, N_img, D] image tokens             txt: [B, N_txt, D], Args:             x: [B, N, D] concatenated [img, txt] tokens             rope:, Get cached K/V for a given layer, or None., Compute rope embeddings for image and text lengths., Compute time + guidance + pooled conditioning., Forward pass.          Args:             img: packed image tokens [B, N_img, 64] (+3 more)

### Community 11 - ".__init__"
Cohesion: 0.18
Nodes (10): DoubleBlock, EmbedND, LinearAttention, Modulation, Modulation layers for diffusion transformer blocks.      Outputs 6 parameters (f, Multi-head attention with QKV projection.      Supports optional Kontext KV cach, FLUX.1 double transformer block.      Processes image and text tokens in paralle, FLUX.1 single transformer block.      Concatenates image and text tokens, then a (+2 more)

### Community 12 - "FluxConfig"
Cohesion: 0.16
Nodes (8): _model_type_from_path(), Diffusion Model Loader ====================== Loads FLUX.1 checkpoints into MLX-, Infer model type from filename., FluxConfig, Configuration module for the native MLX transformer.  Centralizes hyperparameter, FLUX.1 model architecture configuration.      Attributes:         num_double_blo, Dimension per attention head., Validate configuration consistency.

### Community 13 - "ASDX_CheckpointLoader"
Cohesion: 0.15
Nodes (9): ASDX_CheckpointLoader, ASDX_DiffusionLoader, Path, Resolve model name to a file path., Load a full checkpoint (VAE + CLIP + Diffusion) into MLX.      Reads the checkpo, Get list of available checkpoint files., Resolve checkpoint name to a file path., Load a FLUX.1 diffusion model checkpoint into MLX.      Reads the checkpoint, cr (+1 more)

### Community 14 - "native/__init__.py"
Cohesion: 0.19
Nodes (14): _assign_double_block(), _assign_single_block(), _assign_weights(), _load_safetensors(), load_transformer(), _normalize_key(), Path, Native MLX Transformer for FLUX.1 ================================== A minimal b (+6 more)

### Community 15 - "FluxTransformer"
Cohesion: 0.20
Nodes (6): FluxTransformer, Complete FLUX.1 transformer.      Architecture:       img_in / txt_in  ->  19x D, Enable/disable Kontext KV cache., Store reference K/V for a given layer., Convenience method for one denoising step.          Args:             img: [B, N, MLX Native Sampler ================== Euler sampler running entirely in MLX on A

### Community 16 - "ASDX_ApplyControlNet"
Cohesion: 0.32
Nodes (5): ASDX_ApplyControlNet, Tensor, Apply ControlNet conditioning to a diffusion model.      Injects ControlNet resi, Attach ControlNet conditioning to the model., Convert image (+ optional mask) to MLX array [B, 4, H, W].

### Community 17 - "apply_rope"
Cohesion: 0.50
Nodes (3): apply_rope(), Args:             x: [B, N, D] input             rope: [N, D] rope embeddings, Apply rotary positional embeddings to QK pairs.

## Knowledge Gaps
- **14 isolated node(s):** `VAE (encode/decode)`, `T5-XXL`, `Diffusion sampler (Euler)`, `LoRA Schedule`, `Apple Silicon (M1-M5)` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FluxConfig` connect `FluxConfig` to `.mlx_dtype`, `.__init__`, `native/__init__.py`, `FluxTransformer`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Why does `FluxTransformer` connect `FluxTransformer` to `.sample`, `array`, `.__init__`, `FluxConfig`, `native/__init__.py`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `get_vae_encoder()` connect `vae.py` to `mlx_vae.py`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `FluxConfig` (e.g. with `DoubleBlock` and `EmbedND`) actually correct?**
  _`FluxConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `VAE (encode/decode)`, `T5-XXL`, `Diffusion sampler (Euler)` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `controlnet.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06938020351526364 - nodes in this community are weakly interconnected._
- **Should `ASDX - Apple Silicon Diffusion Nodes` be split into smaller, more focused modules?**
  _Cohesion score 0.06923076923076923 - nodes in this community are weakly interconnected._