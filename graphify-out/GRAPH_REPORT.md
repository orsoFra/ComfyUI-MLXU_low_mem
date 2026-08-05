# Graph Report - .  (2026-08-05)

## Corpus Check
- 65 files · ~62,034 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1109 nodes · 1917 edges · 60 communities (58 shown, 2 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 88 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- PyTorch/MLX Bridge Layer
- SDXL Config
- Z-Image Config
- MLX VAE (Placeholder)
- Capability Profiles
- IP-Adapter Nodes
- Flow Sampler & Flux Latent
- LoRA Loader/Schedule
- Krea2 Attention/QKNorm/RMSNorm
- Sampler Core Rationale
- CLIP Loader & Text Encode
- Flux2 Config & Init
- Krea2 Attention Forward
- Krea2 Weight Loading
- Flux2 DoubleBlock/LastLayer
- ControlNet Flux Model
- Live Preview Registry
- FLUX DoubleBlock/Modulation
- Krea2 RoPE (3-axis)
- comfy-reference-diff Agent Checklist
- FLUX Config
- Krea2T Enhancer
- ComfyUI Node Conventions Docs
- Depth Map Node
- Flux2 Joint Attention
- FLUX Joint Attention
- Krea2 LastLayer/SimpleModulation
- Memory/Cache Manager
- Sampler Core & Wan21 Latent
- FLUX Weight Loading
- Krea2 Config
- add-model-family Skill
- SDXL Sampling Schedule
- verify-checkpoint Script Template
- Flux2 Config Dataclass
- Flux2Transformer Forward
- Sampler Core img2img/Inpaint
- FLUX Sigma Schedule
- FLUX Sampler Sigma
- README Node Docs
- Image-to-Latent & Mask Blur
- Generation Metadata
- Image Compositor/Mask
- TeaCache State
- ASDX_MLXSampler Node
- weight-map-reviewer Agent Checklist
- MPS->MLX Conversion Docs
- ControlNet Apply Node
- MFluxImage Dataclass
- Empty FLUX Latent
- ControlNet Union Loader
- FluxTransformer Forward
- Krea2 TextFusion Blocks
- Node Registry Fallback
- Multi-Model Plan Vision
- Flux2 RoPE Embed
- FLUX RoPE Embed
- Krea2 DoubleSharedModulation
- Flux2 Modulation
- Precision/Memory Policy Notes

## God Nodes (most connected - your core abstractions)
1. `Krea2Config` - 27 edges
2. `_SamplerCore` - 26 edges
3. `FluxConfig` - 22 edges
4. `SDXLConfig` - 20 edges
5. `Flux2Config` - 19 edges
6. `ZImageConfig` - 19 edges
7. `load_krea2_transformer()` - 17 edges
8. `_load_safetensors()` - 16 edges
9. `RMSNorm` - 16 edges
10. `SingleStreamDiT` - 15 edges

## Surprising Connections (you probably didn't know these)
- `MLX vs PyTorch MPS Framework Strategy` --semantically_similar_to--> `Plan de Conversion MPS vers MLX Natif`  [INFERRED] [semantically similar]
  CLAUDE.md → docs/conversion-mps-mlx.md
- `ASDX_MLXSampler Node` --semantically_similar_to--> `ASDX_MLXSampler Parameters`  [INFERRED] [semantically similar]
  README.md → docs/nodes-reference.md
- `ComfyUI Custom Node Structure Standard` --semantically_similar_to--> `Complete Node Reference Table`  [INFERRED] [semantically similar]
  CLAUDE.md → docs/nodes-reference.md
- `ASDX - Apple Silicon Diffusion Nodes Project` --semantically_similar_to--> `Multi-Model Apple Silicon Platform Vision`  [INFERRED] [semantically similar]
  README.md → docs/plan-multi-modeles-apple-silicon.md
- `FluxConfig Identified as Most-Connected Graph Node` --semantically_similar_to--> `Target native/ Directory Architecture`  [INFERRED] [semantically similar]
  session-state.md → docs/plan-multi-modeles-apple-silicon.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **7-Point Architecture Review Checklist** — claude_agents_comfy_reference_diff_layer_inventory_check, claude_agents_comfy_reference_diff_shapes_constants_check, claude_agents_comfy_reference_diff_bias_check, claude_agents_comfy_reference_diff_normalization_check, claude_agents_comfy_reference_diff_modulation_check, claude_agents_comfy_reference_diff_sign_convention_check, claude_agents_comfy_reference_diff_rope_convention_check [INFERRED 0.85]
- **6-Point Checkpoint-Loading Review Checklist** — claude_agents_weight_map_reviewer_key_comparison_check, claude_agents_weight_map_reviewer_matched_count_check, claude_agents_weight_map_reviewer_normalize_map_ordering_check, claude_agents_weight_map_reviewer_dtype_check, claude_agents_weight_map_reviewer_tree_unflatten_sequencing_check, claude_agents_weight_map_reviewer_unused_missing_key_check [INFERRED 0.85]
- **4-Step Checkpoint Verification Recipe** — claude_skills_verify_checkpoint_step1_pycompile, claude_skills_verify_checkpoint_step2_forward_pass, claude_skills_verify_checkpoint_step3_matched_load, claude_skills_verify_checkpoint_step4_std_sanity [EXTRACTED 1.00]

## Communities (60 total, 2 thin omitted)

### Community 0 - "PyTorch/MLX Bridge Layer"
Cohesion: 0.06
Nodes (70): clear_mlx_cache(), collect_mlx_memory(), conditioning_flux2_to_mlx(), conditioning_sdxl_to_mlx(), conditioning_to_mlx(), conditioning_zimage_to_mlx(), mlx_to_comfy_image(), mlx_to_comfy_latent() (+62 more)

### Community 1 - "SDXL Config"
Cohesion: 0.05
Nodes (43): process_sdxl_latent_in(), process_sdxl_latent_out(), Any, Dtype, SDXL (UNetModel) architecture configuration.  Covers SDXL base and its checkpoin, Process latent for model input: latent * scale., Process latent for model output: latent / scale., SDXL UNet architecture configuration.      Matches comfy's SDXL unet_config (`co (+35 more)

### Community 2 - "Z-Image Config"
Cohesion: 0.06
Nodes (42): Dtype, Z-Image (NextDiT / Lumina2 family) model configuration.  Values confirmed agains, Z-Image (NextDiT) architecture configuration.      Attributes:         dim: Hidd, Convert dtype string to mlx.core dtype., Dimension per attention head (128 for Z-Image)., FFN hidden dim: multiple_of * ceil(ffn_dim_multiplier * dim / multiple_of)., Validate configuration consistency., ZImageConfig (+34 more)

### Community 3 - "MLX VAE (Placeholder)"
Cohesion: 0.06
Nodes (32): get_vae_decoder(), get_vae_encoder(), GroupNorm, _make_down_block(), _make_up_block(), array, MLX VAE Decoder/Encoder ======================== Lightweight FLUX VAE implementa, Get the cached VAE decoder instance. (+24 more)

### Community 4 - "Capability Profiles"
Cohesion: 0.07
Nodes (38): CapabilityProfile, filter_params_for_model(), Any, Path, Capability profiles for mflux-AnyModel-inspired model dispatch.  Each diffusion, Describes what parameters a model family supports.      Mirrors the CapabilityPr, Resolve a model name to its CapabilityProfile.      Uses the alias dispatch tabl, Resolve capability from a file path (extracts basename). (+30 more)

### Community 5 - "IP-Adapter Nodes"
Cohesion: 0.07
Nodes (22): ASDX_ApplyIPAdapter, ASDX_IPAdapterCLIPVisionEncode, ASDX_IPAdapterLoader, CLIPVisionEncoder, IPAdapterCache, Any, array, Path (+14 more)

### Community 6 - "Flow Sampler & Flux Latent"
Cohesion: 0.06
Nodes (18): FlowSampler, FluxLatentFormat, Krea2Sampler, Any, Calculate denoised output from model prediction.          Computes: model_input, Sigma scheduling for discrete flow matching models.      Similar to FluxSampler, Sampler for Krea2 (SingleStreamDiT) flow-matching models.      Krea2 uses a line, Convert timestep to sigma for Krea2.          Krea2 uses a linear schedule: sigm (+10 more)

### Community 7 - "LoRA Loader/Schedule"
Cohesion: 0.09
Nodes (21): ASDX_LoraLoader, ASDX_LoraSchedule, ASDX_MultiLoraLoader, LoRAAdapter, LoRATarget, Any, Path, LoRA Runtime Loading ==================== Load and apply LoRA adapters to FLUX t (+13 more)

### Community 8 - "Krea2 Attention/QKNorm/RMSNorm"
Cohesion: 0.13
Nodes (15): Attention, QKNorm, RMSNorm with the (1 + scale) weight convention.      The stored scale is zero-ce, Per-head Q/K normalization for attention.      Applies separate RMSNorm to Q and, SwiGLU MLP: down(silu(gate) * up).      MLP dimension: ceil((2/3 * features) * m, GQA attention with per-head QK norm and sigmoid-gated output.      Architecture:, RMSNorm, SwiGLU (+7 more)

### Community 9 - "Sampler Core Rationale"
Cohesion: 0.14
Nodes (16): array, Dtype, Execute the MLX-native sampling loop and return the result latent., Run the Z-Image (NextDiT) sampling loop.          Flow-matching, same Euler upda, Run the Flux2/Klein sampling loop.          Flow-matching, same Euler update as, Core denoising loop and acceleration logic.      This class contains the samplin, Check if TeaCache allows skipping. Returns (skip, reason)., Pack the Kontext reference latent into raw 2x2-patch space.          Same patchi (+8 more)

### Community 10 - "CLIP Loader & Text Encode"
Cohesion: 0.10
Nodes (13): ASDX_CLIPLoader, ASDX_CLIPTextEncode, ASDX_ConditioningMerger, ASDX_DualCLIPLoader, _clip_type_from_string(), Any, Conditioning nodes ================== CLIP text encoding and conditioning manipu, Encode text prompts to conditioning for any model type.      Auto-detects FLUX v (+5 more)

### Community 11 - "Flux2 Config & Init"
Cohesion: 0.15
Nodes (18): detect_flux2_config(), process_flux2_latent_in(), process_flux2_latent_out(), Any, Flux2 (FLUX.2/Klein) model configuration.  Defaults verified against a real chec, Process latent for model input: (latent - shift) * scale. No-op for Flux2., Process latent for model output: latent / scale + shift. No-op for Flux2., Derive a Flux2Config from a (normalized, mapped) checkpoint state dict.      Mir (+10 more)

### Community 12 - "Krea2 Attention Forward"
Cohesion: 0.15
Nodes (11): apply_rope(), array, Complete Krea2 SingleStreamDiT transformer.      Architecture:       img_in (lat, Compute timestep embedding and its 6-param modulation projection.          Match, Precompute the RoPE rotation table for a proper 2D image grid.          Matches, Build [h*w, 3] position indices (frame, row, col) for a 2D grid., Unpack context from [B, seq, txtlayers*txtdim] to [B, seq, txtlayers, txtdim]., Forward pass through Krea2 transformer.          Sequence layout matches the Com (+3 more)

### Community 13 - "Krea2 Weight Loading"
Cohesion: 0.13
Nodes (15): load_krea2_transformer(), Load a Krea2 checkpoint into a SingleStreamDiT.      Args:         path: Path to, map_krea2_to_native(), normalize_krea2_keys(), array, State dict weight mapping for Krea2 checkpoints.  Krea2 uses a different key nam, Normalize Krea2 checkpoint keys by stripping common prefixes.      Args:, Map Krea2 checkpoint keys to native naming convention.      Key mappings: (+7 more)

### Community 14 - "Flux2 DoubleBlock/LastLayer"
Cohesion: 0.14
Nodes (11): DoubleBlock, LastLayer, MLPEmbedder, QKNorm, in_dim -> hidden_dim -> SiLU -> hidden_dim. Bias-free for Flux2 (ops_bias=False), Flux2 double-stream block.      Unlike FLUX.1's DoubleBlock, this owns NO Modula, Flux2 single-stream block. Owns no Modulation of its own (see DoubleBlock)., Final output layer: adaLN modulation (scale+shift only) -> LayerNorm -> Linear. (+3 more)

### Community 15 - "ControlNet Flux Model"
Cohesion: 0.15
Nodes (13): ControlNet Union Support ======================== MLX-native ControlNet Union fo, ControlNetFlux, load_controlnet_union(), array, Path, ControlNet Union model for FLUX.1, and weight loading.  Matches the reference ar, Forward pass.          Returns:             {"input": [...N double residuals...], Load a ControlNet Union model for FLUX from a checkpoint file. (+5 more)

### Community 16 - "Live Preview Registry"
Cohesion: 0.11
Nodes (11): ASDX_LivePreview, LivePreviewRegistry, array, Live preview registry for step-by-step sampling feedback.  Mirrors the live prev, Registry for step-by-step preview callbacks.      After each denoising step, the, Register a callback to be called on each step., Remove a previously registered callback., Fire all registered callbacks with current sampling state. (+3 more)

### Community 17 - "FLUX DoubleBlock/Modulation"
Cohesion: 0.15
Nodes (9): LastLayer, Modulation, QKNorm, Per-head RMSNorm applied to Q and K after the head split., adaLN-style modulation: SiLU(vec) -> Linear -> chunk into ModulationOut(s)., Args: vec [B, dim] conditioning vector.          Returns: (mod1, mod2) where eac, QKV + QKNorm + output projection, without running attention itself.      Matches, Final output layer: adaLN modulation -> LayerNorm -> Linear. (+1 more)

### Community 18 - "Krea2 RoPE (3-axis)"
Cohesion: 0.16
Nodes (14): Krea2 (SingleStreamDiT) native MLX implementation.  Provides a complete MLX impl, apply_rope_3d(), compute_rope_3d(), EmbedND, array, Dtype, 3-axis Rotary Positional Embeddings (RoPE) for Krea2 Identity Edit.  Krea2 uses, N-dimensional RoPE embedding module.      Generalizes the standard EmbedND to su (+6 more)

### Community 19 - "comfy-reference-diff Agent Checklist"
Cohesion: 0.12
Nodes (18): comfy-reference-diff Agent, Bias Presence/Absence Check, dynamic-splashing-boot.md Plan Reference, Kontext Single-Reference Scope Exception, Layer Inventory and Order Check, Modulation/Conditioning Mechanics Check, Normalization Placement and Type Check, RoPE/Positional Convention Check (+10 more)

### Community 20 - "FLUX Config"
Cohesion: 0.14
Nodes (10): FluxConfig, Dtype, FLUX.1 model architecture configuration.      Attributes:         num_double_blo, Convert dtype string to mlx.core dtype., Dimension per attention head., Validate configuration consistency., MLPEmbedder, FLUX.1 single transformer block.      Operates on the concatenated [txt, img] se (+2 more)

### Community 21 - "Krea2T Enhancer"
Cohesion: 0.14
Nodes (14): EmbedND, krea2t_enhance_conditioning(), _krea2t_run_txtfusion_parts(), Krea2 SingleStreamDiT transformer model.  Implements the core architecture compo, # NOTE: reshape must first split the last dim (B,L,H,D), then transpose to move, Mirrors `TextFusionTransformer.__call__` exactly (layerwise_blocks ->     projec, Apply the Krea2T prompt-adherence enhancer, then run txtfusion.      Args:, # NOTE: divide by half_dim, NOT half_dim-1 -- comfy's own formula is (+6 more)

### Community 22 - "ComfyUI Node Conventions Docs"
Cohesion: 0.14
Nodes (16): ComfyUI Custom Node Structure Standard, 4-Point Integration Checklist, ComfyUI Tensor Format Convention [B,H,W,C], Capability Profiles Table, Complete Node Reference Table, Expected ComfyUI Model Folder Conventions, ASDX_MLXSampler Parameters, Sampling Modes (text2img/img2img/inpaint/fill/depth) (+8 more)

### Community 23 - "Depth Map Node"
Cohesion: 0.24
Nodes (9): ASDX_DepthMap, Any, Tensor, Generate a simple depth approximation from luminance.          This is a rough a, Convert a torch tensor to PIL Image., Generate a depth map from an RGB image.      Uses a pre-trained depth estimation, Generate depth map from image.          Returns a MFLUX_IMAGE with the depth map, Run depth estimation on the image.          Tries multiple backends in order: (+1 more)

### Community 24 - "Flux2 Joint Attention"
Cohesion: 0.22
Nodes (11): apply_mod(), apply_rope(), joint_attention(), array, x * (1 + scale) [+ shift]. scale/shift are [B,1,D], x is [B,N,D]., [B,N,2*H] fused gate+up projection -> silu(gate) * up -> [B,N,H]., Args:             img: [B, N_img, D]             txt: [B, N_txt, D], Args:             x: [B, N, D] concatenated [txt, img] tokens.             mod: (+3 more)

### Community 25 - "FLUX Joint Attention"
Cohesion: 0.21
Nodes (11): apply_mod(), apply_rope(), DoubleBlock, joint_attention(), array, Apply paired-interleave RoPE rotation to Q or K.      Args:         x: [B, H, N,, x * (1 + scale) [+ shift]. scale/shift are [B, 1, D], x is [B, N, D]., Scaled dot-product attention with RoPE.      Args:         q, k, v: [B, H, N, he (+3 more)

### Community 26 - "Krea2 LastLayer/SimpleModulation"
Cohesion: 0.13
Nodes (9): LastLayer, Timestep modulation: vec + lin → 2 params (scale, shift).      Used in the LastL, Compute scale and shift from timestep embedding.          Args:             vec:, Krea2 single-stream transformer block.      Architecture:       Modulation: Doub, Forward pass through single-stream block.          Args:             x: Input te, Final layer: RMSNorm + SimpleModulation + linear.      Applies timestep-based sc, Forward pass through last layer.          Args:             x: Input tensor [B,, SimpleModulation (+1 more)

### Community 27 - "Memory/Cache Manager"
Cohesion: 0.16
Nodes (7): ASDX_CacheManager, ASDX_MemoryProfiler, Memory Profiler & Cache Manager ================================ Nodes for monit, Collect PyTorch MPS memory statistics., Clear MLX and PyTorch MPS caches.      Use between major workflow phases (e.g.,, Profile and display Apple Silicon Unified Memory usage.      Reports:       - ML, Collect MLX memory statistics.

### Community 28 - "Sampler Core & Wan21 Latent"
Cohesion: 0.15
Nodes (10): process_wan21_latent_in(), Configuration module for the native MLX transformer.  Centralizes hyperparameter, Process latent for model input: (latent - mean) / std, per-channel.      Args:, encode_adm(), Build the SDXL ADM/"y" conditioning vector.      Matches `comfy/model_base.py::S, Caching mechanisms for the MLX sampler.  Contains:   - TeaCacheState: output-lev, Core sampling logic for the MLX-native FLUX sampler.  Contains the denoising loo, Sampling mode: text2img, img2img, inpainting, fill, depth control. (+2 more)

### Community 29 - "FLUX Weight Loading"
Cohesion: 0.22
Nodes (12): _load_safetensors(), load_transformer(), Path, Native MLX Transformer for FLUX.1 ================================== A FLUX.1 tr, Load a safetensors file into MLX arrays.      Uses safetensors.torch to support, Load a FLUX.1 checkpoint into a FluxTransformer.      Args:         path: path t, map_flux_to_native(), normalize_flux_keys() (+4 more)

### Community 30 - "Krea2 Config"
Cohesion: 0.15
Nodes (8): Krea2Config, Dtype, Krea2 (SingleStreamDiT) model configuration.  Defines architecture parameters fo, Krea2 (SingleStreamDiT) model architecture configuration.      Attributes:, Convert dtype string to mlx.core dtype., Dimension per attention head., MLP dimension rounded up to multiple of 128., Validate configuration consistency.

### Community 31 - "add-model-family Skill"
Cohesion: 0.16
Nodes (14): dynamic-splashing-boot.md Plan Reference (Phase B/E Research), native/krea2/ as Config Template, load_<x>_transformer 6-Step Loading Recipe, native/<x>/ Subpackage Skeleton, add-model-family Skill, FLUX.2/Klein No-Real-Checkpoint Gap, Session 11 Checkpoint-Key Bug (Origin of Recipe), verify-checkpoint Skill (+6 more)

### Community 32 - "SDXL Sampling Schedule"
Cohesion: 0.18
Nodes (7): Run the SDXL (conv UNet, EPS/discrete-DDPM) sampling loop.          Fundamentall, generate_sigmas_sdxl(), Discrete DDPM/EPS sigma schedule for SDXL.      Fundamentally different from FLU, Continuous sigma for a (possibly fractional) discrete timestep index., Nearest discrete timestep index for a continuous sigma (log-sigma distance)., Generate an SDXL sigma schedule ('normal'-scheduler style, matching     comfy's, SDXLSampling

### Community 33 - "verify-checkpoint Script Template"
Cohesion: 0.22
Nodes (11): main(), Path, Compare loaded weight std against the characteristic std of MLX's default     nn, Syntax check every file in the family's native/<x>/ subpackage., Forward pass on a REDUCED config with random weights. NaN-free is the bar., # TODO: set reduced-but-valid dims for this family., Load the real checkpoint and report matched/missing/extra key counts.      Uses, step0_py_compile() (+3 more)

### Community 34 - "Flux2 Config Dataclass"
Cohesion: 0.18
Nodes (7): Flux2Config, Dtype, Convert dtype string to mlx.core dtype., Dimension per attention head., MLP hidden dim (down-projection input size): hidden_size * mlp_ratio., Validate configuration consistency., Flux2/Klein model architecture configuration.      Attributes:         num_doubl

### Community 35 - "Flux2Transformer Forward"
Cohesion: 0.18
Nodes (8): Flux2Transformer, Complete Flux2/Klein transformer.      Architecture:       img_in / txt_in -> [d, Compute the 4-axis RoPE table for a [txt, img] sequence.          Image tokens g, [B, hidden_size] conditioning vector: timestep (+ guidance if the         checkp, Forward pass. Returns [B, N_img, in_channels] noise prediction., Convenience method for one denoising step., Sinusoidal timestep embedding, matching comfy.ldm.flux.layers.timestep_embedding, timestep_embedding()

### Community 36 - "Sampler Core img2img/Inpaint"
Cohesion: 0.17
Nodes (7): Any, ndarray, Tensor, Recompute this step's LoRA strength from the schedule and re-apply it., VAE-encode an image to FLUX latent packed format.          Returns MLX packed la, Prepare inpainting noise: encode image, apply mask, add noise.          The mask, Convert mask tensor to packed latent-space mask.          Returns a numpy array

### Community 37 - "FLUX Sigma Schedule"
Cohesion: 0.24
Nodes (11): _flux_fixed_shift_sigmas(), flux_resolution_shift(), flux_time_shift(), generate_sigmas(), Sigma scheduling for diffusion models.  Adapted from DiffusionKit's FluxSampler, Matches comfy.model_sampling.flux_time_shift exactly., Matches comfy.model_sampling.time_snr_shift exactly.      Used by `ModelSampling, Sigma schedule for a `ModelSamplingFlux`-backed model with a fixed     (non-reso (+3 more)

### Community 38 - "FLUX Sampler Sigma"
Cohesion: 0.17
Nodes (7): FluxSampler, Sigma scheduling for FLUX models.      FLUX uses a continuous sigma schedule whe, Minimum sigma value (at timestep 0)., Maximum sigma value (at timestep 1000)., Convert a timestep value to sigma.          Args:             timestep: Timestep, Compute sigma for a specific timestep index., Convert a sigma value to timestep.          Args:             sigma: Sigma value

### Community 39 - "README Node Docs"
Cohesion: 0.21
Nodes (12): ControlNet Union 8 Control Types, ASDX_MLXSampler Node, ASDX - Apple Silicon Diffusion Nodes Project, ASDX VAE Decode (MLX) Node, ControlNet Union Loader / Apply ControlNet Nodes, IP-Adapter Loader / CLIP Vision Encode / Apply IP-Adapter Nodes, LoRA Loader / Multi LoRA / LoRA Schedule Nodes, ComfyUI Pipeline Architecture Diagram (+4 more)

### Community 40 - "Image-to-Latent & Mask Blur"
Cohesion: 0.22
Nodes (6): ASDX_ImageToLatent, ASDX_MaskBlur, Tensor, Apply Gaussian blur to a mask for soft edges., Create a 1D Gaussian kernel, then make it 2D via outer product., VAE-encode an image to latent, return MFLUX_IMAGE.      Encodes a [B, H, W, C] i

### Community 41 - "Generation Metadata"
Cohesion: 0.25
Nodes (10): build_generation_metadata(), extract_png_metadata(), _json_default(), Any, Metadata sidecar saving for generated images.  Adapted from Mflux-ComfyUI's save, Extract metadata from a PNG image's text chunks.      Reads the metadata embedde, Default JSON serializer for unsupported types., Save a JSON sidecar file alongside an output image.      The JSON file has the s (+2 more)

### Community 42 - "Image Compositor/Mask"
Cohesion: 0.20
Nodes (5): ASDX_ImageCompositor, ASDX_MaskFromImage, MFLUX_IMAGE typed chain for multi-image workflows.  Mirrors the MfluxImage patte, Generate a binary mask from an image using threshold.      Converts a grayscale, Composite a generated image over an original using a mask.      Implements mask-

### Community 43 - "TeaCache State"
Cohesion: 0.27
Nodes (6): array, Record a real (non-skipped) step's output., TeaCache state for output-level step skipping.      TeaCache works by comparing, Compute adaptive threshold that interpolates between start and end., Try to reuse previous output. Returns (output, skipped, reason)., TeaCacheState

### Community 44 - "ASDX_MLXSampler Node"
Cohesion: 0.22
Nodes (6): ASDX_MLXSampler, Tensor, MLX Native Sampler ================== Euler sampler running entirely in MLX on A, Execute the MLX-native sampling loop via _SamplerCore., Get latent previewer from ComfyUI., MLX-native FLUX sampler with SeaCache acceleration.      Runs the full denoising

### Community 45 - "weight-map-reviewer Agent Checklist"
Cohesion: 0.22
Nodes (10): weight-map-reviewer Agent, dtype Handling Check, dynamic-splashing-boot.md Pattern a repliquer Reference, Key Comparison Type Mismatch Check, Matched-Count Logging Check, normalize_/map_ Ordering and Idempotence Check, Session 11 Tuple-vs-String Key Comparison Bug, tree_unflatten/update/eval Sequencing Check (+2 more)

### Community 46 - "MPS->MLX Conversion Docs"
Cohesion: 0.24
Nodes (10): native/clip_encoder.py - CLIP-L MLX Implementation, native/depth_encoder.py - Monodepth2/DepthPro Options, ASDX_MLX_ENCODERS Migration Flag, Phase 4 Image Chain MLX - Not Recommended Decision, Phase 5 Live Preview MLX - Not Recommended Decision, ComfyUI-mflux-AnyModel Reference Project, Plan de Conversion MPS vers MLX Natif, native/t5_encoder.py - T5-XXL MLX Implementation (+2 more)

### Community 47 - "ControlNet Apply Node"
Cohesion: 0.28
Nodes (6): ASDX_ApplyControlNet, Any, Tensor, Attach ControlNet conditioning to the model., Concatenate a mask as an extra image channel (inpaint ControlNet-Union)., Apply ControlNet conditioning to a diffusion model.      VAE-encodes the control

### Community 48 - "MFluxImage Dataclass"
Cohesion: 0.22
Nodes (4): MFLUX_IMAGE, Any, Typed image payload for chaining in multi-image workflows.      Combines image,, Serialize for ComfyUI bridge (type hints only, no tensor data).

### Community 49 - "Empty FLUX Latent"
Cohesion: 0.25
Nodes (5): ASDX_EmptyFLUXLatent, Empty FLUX Latent ================= Creates empty 16-channel FLUX latents optimi, Get the best available device., Create an empty 16-channel FLUX latent tensor.      The latent is placed on the, device

### Community 50 - "ControlNet Union Loader"
Cohesion: 0.32
Nodes (4): ASDX_ControlNetUnionLoader, Path, Load a ControlNet Union model., Get list of available ControlNet models.

### Community 51 - "FluxTransformer Forward"
Cohesion: 0.29
Nodes (5): FluxTransformer, Complete FLUX.1 transformer.      Architecture:       img_in / txt_in  ->  19x D, Compute the [B, hidden_dim] conditioning vector: time + guidance + pooled., Forward pass.          Args:             img: packed image tokens [B, N_img, 64], Convenience method for one denoising step.          Args:             img: [B, N

### Community 52 - "Krea2 TextFusion Blocks"
Cohesion: 0.25
Nodes (5): Text fusion block: RMSNorm + attention + SwiGLU MLP.      Uses separate pre and, Text fusion adapter for Qwen3-VL layer taps.      Architecture:       Input:  [B, Forward pass through text fusion transformer.          Args:             x: Unpa, TextFusionBlock, TextFusionTransformer

### Community 53 - "Node Registry Fallback"
Cohesion: 0.29
Nodes (3): Depth map generation using MLX-compatible depth estimation.  Mirrors the MfluxDe, Apple Silicon Diffusion Nodes ============================= Custom ComfyUI nodes, _Unavailable

### Community 54 - "Multi-Model Plan Vision"
Cohesion: 0.33
Nodes (7): MLX vs PyTorch MPS Framework Strategy, PyTorch-MLX Interoperability Bridge, ASDX_MULTI_MODEL Backward-Compatibility Flag, Implementation Phases 0-5, Target Model Families Table, Unified Memory Strategy Rules, Multi-Model Apple Silicon Platform Vision

### Community 55 - "Flux2 RoPE Embed"
Cohesion: 0.40
Nodes (5): embed_nd(), Native MLX Transformer for Flux2 (FLUX.2/Klein) ================================, [N, dim/2, 2, 2] rotation-matrix RoPE table for one axis., N-axis RoPE embedding table (generic over axis count, unlike the name suggests)., rope_freqs()

### Community 56 - "FLUX RoPE Embed"
Cohesion: 0.33
Nodes (5): embed_nd(), 3-axis RoPE embedding table.      Args:         ids: [N, num_axes] position indi, Compute the 3-axis RoPE table for a [txt, img, ref...] sequence.          Matche, Compute a [N, dim/2, 2, 2] rotation-matrix RoPE table for one axis.      Matches, rope_freqs()

### Community 57 - "Krea2 DoubleSharedModulation"
Cohesion: 0.40
Nodes (3): DoubleSharedModulation, Timestep modulation: vec + lin → 6 params (chunk).      Simple additive modulati, Compute 6 modulation parameters from timestep embedding.          Args:

## Knowledge Gaps
- **30 isolated node(s):** `Layer Inventory and Order Check`, `Bias Presence/Absence Check`, `Normalization Placement and Type Check`, `Modulation/Conditioning Mechanics Check`, `dynamic-splashing-boot.md Plan Reference` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_load_safetensors()` connect `FLUX Weight Loading` to `SDXL Config`, `Z-Image Config`, `Capability Profiles`, `Flux2 Config & Init`, `Krea2 Weight Loading`, `ControlNet Flux Model`, `Krea2T Enhancer`, `Flux2 RoPE Embed`, `FLUX Joint Attention`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `load_flux2_transformer()` connect `Flux2 Config & Init` to `Flux2Transformer Forward`, `Capability Profiles`, `FLUX Weight Loading`, `Flux2 RoPE Embed`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `Krea2Config` connect `Krea2 Config` to `Capability Profiles`, `Krea2 Attention/QKNorm/RMSNorm`, `Krea2 Attention Forward`, `Krea2 Weight Loading`, `Krea2 RoPE (3-axis)`, `Krea2 TextFusion Blocks`, `Krea2T Enhancer`, `Krea2 DoubleSharedModulation`, `Krea2 LastLayer/SimpleModulation`, `FLUX Weight Loading`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `Krea2Config` (e.g. with `Attention` and `DoubleSharedModulation`) actually correct?**
  _`Krea2Config` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `_SamplerCore` (e.g. with `TeaCacheState` and `FluxSampler`) actually correct?**
  _`_SamplerCore` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `FluxConfig` (e.g. with `DoubleBlock` and `FluxTransformer`) actually correct?**
  _`FluxConfig` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Layer Inventory and Order Check`, `Bias Presence/Absence Check`, `Normalization Placement and Type Check` to the rest of the system?**
  _30 weakly-connected nodes found - possible documentation gaps or missing edges._