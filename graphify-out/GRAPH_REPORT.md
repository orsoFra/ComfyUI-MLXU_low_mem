# Graph Report - .  (2026-08-02)

## Corpus Check
- Corpus is ~21,835 words - fits in a single context window. You may not need a graph.

## Summary
- 588 nodes · 922 edges · 26 communities
- Extraction: 83% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- ControlNet Blocks & Embedding
- Depth Map Generation
- VAE MLX Encode/Decode
- Package Structure & Entry Point
- IP-Adapter Cross-Attention
- LoRA Runtime Loading
- Graphify Audit & Isolated Nodes
- Module Init & Fallback
- Bridge PyTorch-MLX
- Capability Profiles
- CLIP Conditioning & Text Encode
- Flux Transformer Core
- Live Preview Registry
- DoubleBlock & EmbedND Layers
- Kontext KV Cache
- Sampler Core Logic
- Model Loader & Cache
- FluxConfig
- Sampler Rationales
- Kontext Attention
- Weight Assignment
- Sampler Node Interface
- Apply ControlNet
- Inpainting Noise
- RoPE & LinearAttention
- Canon Protocol

## God Nodes (most connected - your core abstractions)
1. `FluxConfig` - 19 edges
2. `_SamplerCore` - 19 edges
3. `ControlNetUnionModel` - 17 edges
4. `MFLUX_IMAGE` - 14 edges
5. `FluxTransformer` - 14 edges
6. `KontextCache` - 13 edges
7. `load_transformer()` - 11 edges
8. `TeaCacheState` - 11 edges
9. `ASDX_DepthMap` - 9 edges
10. `CapabilityProfile` - 8 edges

## Surprising Connections (you probably didn't know these)
- `ASDX_ApplyControlNet` --uses--> `ControlNetUnionModel`  [INFERRED]
  apple_silicon_nodes/controlnet/__init__.py → apple_silicon_nodes/controlnet/model.py
- `EmbedND` --uses--> `FluxConfig`  [INFERRED]
  apple_silicon_nodes/native/__init__.py → apple_silicon_nodes/native/config.py
- `Modulation` --uses--> `FluxConfig`  [INFERRED]
  apple_silicon_nodes/native/__init__.py → apple_silicon_nodes/native/config.py
- `LinearAttention` --uses--> `FluxConfig`  [INFERRED]
  apple_silicon_nodes/native/__init__.py → apple_silicon_nodes/native/config.py
- `DoubleBlock` --uses--> `FluxConfig`  [INFERRED]
  apple_silicon_nodes/native/__init__.py → apple_silicon_nodes/native/config.py

## Import Cycles
- 3-file cycle: `apple_silicon_nodes/__init__.py -> apple_silicon_nodes/sampler/__init__.py -> apple_silicon_nodes/sampler/core.py -> apple_silicon_nodes/__init__.py`

## Communities (26 total, 0 thin omitted)

### Community 0 - "ControlNet Blocks & Embedding"
Cohesion: 0.06
Nodes (34): ControlNetCondEmbedding, array, ControlNet Union building blocks.  MLX-native layers used by ControlNetUnionMode, Embed control image into conditioning features., Timestep embedding with optional text time augmentation., Sinusoidal positional encoding for timestep embeddings., Compute sinusoidal encoding for timestep array., SinusoidalPositionalEncoding (+26 more)

### Community 1 - "Depth Map Generation"
Cohesion: 0.06
Nodes (25): ASDX_DepthMap, Any, Tensor, Depth map generation using MLX-compatible depth estimation.  Mirrors the MfluxDe, Generate a simple depth approximation from luminance.          This is a rough a, Convert a torch tensor to PIL Image., Generate a depth map from an RGB image.      Uses a pre-trained depth estimation, Generate depth map from image.          Returns a MFLUX_IMAGE with the depth map (+17 more)

### Community 2 - "VAE MLX Encode/Decode"
Cohesion: 0.07
Nodes (30): get_vae_decoder(), get_vae_encoder(), GroupNorm, _make_down_block(), _make_up_block(), array, MLX VAE Decoder/Encoder ======================== Lightweight FLUX VAE implementa, Get the cached VAE decoder instance. (+22 more)

### Community 3 - "Package Structure & Entry Point"
Cohesion: 0.08
Nodes (45): apple_silicon_nodes, apple_silicon_nodes/controlnet/, apple_silicon_nodes/__init__.py, apple_silicon_nodes/loader.py, apple_silicon_nodes/sampler/, Pipeline Architecture, PyTorch-MLX Bridge, Cache Management (+37 more)

### Community 4 - "IP-Adapter Cross-Attention"
Cohesion: 0.07
Nodes (22): ASDX_ApplyIPAdapter, ASDX_IPAdapterCLIPVisionEncode, ASDX_IPAdapterLoader, CLIPVisionEncoder, IPAdapterCache, Any, array, Path (+14 more)

### Community 5 - "LoRA Runtime Loading"
Cohesion: 0.08
Nodes (22): ASDX_LoraLoader, ASDX_LoraSchedule, ASDX_MultiLoraLoader, LoRAAdapter, LoRATarget, Any, Path, LoRA Runtime Loading ==================== Load and apply LoRA adapters to FLUX t (+14 more)

### Community 6 - "Graphify Audit & Isolated Nodes"
Cohesion: 0.11
Nodes (29): 14 Isolated Nodes, CLIP-Vision-H, ControlNet Control Types, ControlNet Union, Euler Sampler, FluxConfig, FLUX Transformer, FLUX.2 / Klein (+21 more)

### Community 7 - "Module Init & Fallback"
Cohesion: 0.08
Nodes (14): Apple Silicon Diffusion Nodes ============================= Custom ComfyUI nodes, _Unavailable, ASDX_EmptyFLUXLatent, Empty FLUX Latent ================= Creates empty 16-channel FLUX latents optimi, Create an empty 16-channel FLUX latent tensor.      The latent is placed on the, Get the best available device., ASDX_CacheManager, ASDX_MemoryProfiler (+6 more)

### Community 8 - "Bridge PyTorch-MLX"
Cohesion: 0.11
Nodes (26): clear_mlx_cache(), collect_mlx_memory(), conditioning_to_mlx(), mlx_to_comfy_image(), mlx_to_comfy_latent(), prepare_noise_from_latent(), Any, array (+18 more)

### Community 9 - "Capability Profiles"
Cohesion: 0.09
Nodes (20): CapabilityProfile, filter_params_for_model(), Any, Path, Capability profiles for mflux-AnyModel-inspired model dispatch.  Each diffusion, Resolve a model name to its CapabilityProfile.      Uses the alias dispatch tabl, Resolve capability from a file path (extracts basename)., Filter candidate params according to a CapabilityProfile.      Returns (valid_pa (+12 more)

### Community 10 - "CLIP Conditioning & Text Encode"
Cohesion: 0.10
Nodes (13): ASDX_CLIPLoader, ASDX_CLIPTextEncode, ASDX_ConditioningMerger, ASDX_DualCLIPLoader, _clip_type_from_string(), Any, Conditioning nodes ================== CLIP text encoding and conditioning manipu, Encode text prompts to conditioning for any model type.      Auto-detects FLUX v (+5 more)

### Community 11 - "Flux Transformer Core"
Cohesion: 0.10
Nodes (16): FluxTransformer, array, Returns a tuple of (param_i * x) for each of num_params., Args:             img: [B, N_img, D] image tokens             txt: [B, N_txt, D], Args:             x: [B, N, D] concatenated [img, txt] tokens             rope:, Complete FLUX.1 transformer.      Architecture:       img_in / txt_in  ->  19x D, Enable/disable Kontext KV cache., Get cached K/V for a given layer, or None. (+8 more)

### Community 12 - "Live Preview Registry"
Cohesion: 0.11
Nodes (11): ASDX_LivePreview, LivePreviewRegistry, array, Live preview registry for step-by-step sampling feedback.  Mirrors the live prev, Registry for step-by-step preview callbacks.      After each denoising step, the, Register a callback to be called on each step., Remove a previously registered callback., Fire all registered callbacks with current sampling state. (+3 more)

### Community 13 - "DoubleBlock & EmbedND Layers"
Cohesion: 0.18
Nodes (10): DoubleBlock, EmbedND, LinearAttention, Modulation, Modulation layers for diffusion transformer blocks.      Outputs 6 parameters (f, Multi-head attention with QKV projection.      Supports optional Kontext KV cach, FLUX.1 double transformer block.      Processes image and text tokens in paralle, FLUX.1 single transformer block.      Concatenates image and text tokens, then a (+2 more)

### Community 14 - "Kontext KV Cache"
Cohesion: 0.18
Nodes (8): KontextCache, Caching mechanisms for the MLX sampler.  Contains:   - TeaCacheState: output-lev, KV cache for reference image tokens in transformer attention.      Caches K/V pa, TeaCache state for output-level step skipping.      TeaCache works by comparing, TeaCacheState, Core sampling logic for the MLX-native FLUX sampler.  Contains the denoising loo, Sampling mode: text2img, img2img, inpainting, fill, depth control., SamplerMode

### Community 15 - "Sampler Core Logic"
Cohesion: 0.17
Nodes (9): Any, array, Dtype, Tensor, Pre-compute reference latent encoding for Kontext conditioning., Convert current MLX latent to a decodable PyTorch latent for preview., VAE-encode an image to FLUX latent packed format.          Returns MLX packed la, Encode input image, add noise at specified strength level.          Returns MLX (+1 more)

### Community 16 - "Model Loader & Cache"
Cohesion: 0.21
Nodes (10): _build_cache_key(), _model_type_from_path(), Diffusion Model Loader ====================== Loads FLUX.1 checkpoints into MLX-, Build a composite cache key matching mflux-AnyModel pattern.      Combines the b, Infer model type from filename., _load_safetensors(), load_transformer(), Path (+2 more)

### Community 17 - "FluxConfig"
Cohesion: 0.18
Nodes (7): FluxConfig, Dtype, Configuration module for the native MLX transformer.  Centralizes hyperparameter, FLUX.1 model architecture configuration.      Attributes:         num_double_blo, Convert dtype string to mlx.core dtype., Dimension per attention head., Validate configuration consistency.

### Community 18 - "Sampler Rationales"
Cohesion: 0.23
Nodes (7): Execute the MLX-native sampling loop and return the result latent., Check if TeaCache allows skipping. Returns (skip, reason)., Core denoising loop and acceleration logic.      This class contains the samplin, Update LoRA strength based on schedule., Create sigma schedule for the given model type.          FLUX dev uses a shifted, Detect sampling mode from connected inputs.          Priority: inpainting (image, _SamplerCore

### Community 19 - "Kontext Attention"
Cohesion: 0.22
Nodes (6): array, Record a real (non-skipped) step's output., Get attention output, using cached reference K/V if available., Standard scaled dot-product attention., Compute adaptive threshold that interpolates between start and end., Try to reuse previous output. Returns (output, skipped, reason).

### Community 20 - "Weight Assignment"
Cohesion: 0.24
Nodes (9): _assign_double_block(), _assign_single_block(), _assign_weights(), _normalize_key(), Native MLX Transformer for FLUX.1 ================================== A minimal b, Normalize a PyTorch/ComfyUI key to our internal naming., Assign loaded weights to model parameters., Assign weights to a DoubleBlock. (+1 more)

### Community 21 - "Sampler Node Interface"
Cohesion: 0.22
Nodes (6): ASDX_MLXSampler, Tensor, MLX Native Sampler ================== Euler sampler running entirely in MLX on A, Execute the MLX-native sampling loop via _SamplerCore., Get latent previewer from ComfyUI., MLX-native FLUX sampler with SeaCache acceleration.      Runs the full denoising

### Community 22 - "Apply ControlNet"
Cohesion: 0.28
Nodes (6): ASDX_ApplyControlNet, array, Tensor, Attach ControlNet conditioning to the model., Convert image (+ optional mask) to MLX array [B, 4, H, W]., Apply ControlNet conditioning to a diffusion model.      Injects ControlNet resi

### Community 23 - "Inpainting Noise"
Cohesion: 0.40
Nodes (3): ndarray, Prepare inpainting noise: encode image, apply mask, add noise.          The mask, Convert mask tensor to packed latent-space mask.          Returns a numpy array

### Community 24 - "RoPE & LinearAttention"
Cohesion: 0.50
Nodes (3): apply_rope(), Args:             x: [B, N, D] input             rope: [N, D] rope embeddings, Apply rotary positional embeddings to QK pairs.

### Community 25 - "Canon Protocol"
Cohesion: 0.50
Nodes (4): Canon Protocol, Project Canon System, Record Kinds, Record Status

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FluxConfig` connect `FluxConfig` to `Model Loader & Cache`, `Flux Transformer Core`, `Weight Assignment`, `DoubleBlock & EmbedND Layers`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `_SamplerCore` connect `Sampler Rationales` to `Inpainting Noise`, `Sampler Node Interface`, `Kontext KV Cache`, `Sampler Core Logic`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `ControlNetUnionModel` connect `ControlNet Blocks & Embedding` to `Apply ControlNet`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `FluxConfig` (e.g. with `DoubleBlock` and `EmbedND`) actually correct?**
  _`FluxConfig` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_SamplerCore` (e.g. with `KontextCache` and `TeaCacheState`) actually correct?**
  _`_SamplerCore` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `ControlNetUnionModel` (e.g. with `ASDX_ApplyControlNet` and `ASDX_ControlNetUnionLoader`) actually correct?**
  _`ControlNetUnionModel` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `ControlNet Blocks & Embedding` be split into smaller, more focused modules?**
  _Cohesion score 0.06060606060606061 - nodes in this community are weakly interconnected._