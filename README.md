# ASDX — Apple Silicon Diffusion Nodes

Custom ComfyUI nodes optimized for **Apple Silicon** (M1/M2/M3/M4/M5) using **MLX native inference** with zero-copy Unified Memory semantics.

Inspired by [SDMLX](https://github.com/elef4/SDMLX), this project takes the core concepts and reimagines them with a cleaner architecture focused on:

1. **Pure MLX inference** — the diffusion loop runs entirely in MLX, no PyTorch overhead
2. **Strategic memory management** — `mx.eval()` at bridge points, `mx.clear_cache()` between phases
3. **Precision awareness** — float16/bfloat16 native on Apple Silicon
4. **Clean separation** — bridge, native core, and nodes are independently maintainable

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ComfyUI Pipeline                                  │
└───────────────┬───────────────────────────┬─────────────────────────┘
                │                           │
     ┌──────────▼──────────┐    ┌──────────▼──────────┐
     │   Conditioning      │    │    Latent Input      │
     │   (CLIP/T5)         │    │    (16-channel)      │
     └──────────┬──────────┘    └──────────┬──────────┘
                │                          │
     ┌──────────▼──────────┐    ┌──────────▼──────────┐
     │  mlx_conditioning   │    │  torch.Tensor (MPS) │
     └──────────┬──────────┘    └──────────┬──────────┘
                │                          │
                └──────────┬───────────────┘
                           │
              ┌────────────▼────────────┐
              │   ASDX_MLXSampler       │
              │   (Pure MLX inference)  │
              │                         │
              │  • FluxTransformer      │
              │  • SeaCache / TeaCache  │
              │  • Kontext KV cache     │
              │  • LoRA injection       │
              │  • ControlNet conditioning│
              │  • IP-Adapter injection │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  mx.array (latent)      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   ASDX_VAEDecode        │
              │   (MLX VAE decode)      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  torch.Tensor (IMAGE)   │
              │  [B, H, W, C] [0,1]     │
              └─────────────────────────┘
```

## Nodes

### Loaders
| Node | Description |
|------|-------------|
| `🍏 ASDX Diffusion Loader` | Load FLUX.1 checkpoint into MLX transformer |
| `🍏 ASDX Dual CLIP Loader` | Load CLIP-L + T5-XXL text encoders |

### Conditioning
| Node | Description |
|------|-------------|
| `🍏 ASDX CLIP Text Encode FLUX` | Encode prompts to T5 + CLIP embeddings |
| `🍏 ASDX Conditioning Merger` | Merge conditioning inputs |

### Sampling
| Node | Description |
|------|-------------|
| `🍏 ASDX MLX Native Sampler` | Euler sampler in pure MLX with SeaCache |

### Latent
| Node | Description |
|------|-------------|
| `🍏 ASDX Empty FLUX Latent` | Create 16-channel FLUX latent |
| `🍏 ASDX VAE Decode (MLX)` | Decode latents via MLX VAE |
| `🍏 ASDX VAE Encode (MLX)` | Encode images via MLX VAE |

### LoRA
| Node | Description |
|------|-------------|
| `🍏 ASDX LoRA Loader` | Load single LoRA (A/B or ComfyUI diff format) with strength scaling |
| `🍏 ASDX Multi LoRA Loader` | Stack up to 5 LoRAs simultaneously |
| `🍏 ASDX LoRA Schedule` | Per-step LoRA strength modulation (linear/cosine/ease-in-out) |

### ControlNet
| Node | Description |
|------|-------------|
| `🍏 ASDX ControlNet Union Loader` | Load ControlNet supporting 8 control types |
| `🍏 ASDX Apply ControlNet` | Apply control conditioning to the diffusion process |

### IP-Adapter
| Node | Description |
|------|-------------|
| `🍏 ASDX IP-Adapter Loader` | Load IP-Adapter projection weights |
| `🍏 ASDX CLIP Vision Encode` | Encode reference image to CLIP-Vision tokens |
| `🍏 ASDX Apply IP-Adapter` | Inject reference image features via cross-attention |

### Utilities
| Node | Description |
|------|-------------|
| `🍏 ASDX Memory Profiler` | Real-time MLX/MPS memory stats |
| `🍏 ASDX Cache Manager` | Clear caches between phases |

## Installation

1. Place the `apple_silicon_nodes` folder into your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ln -s /path/to/ComfyUI-MLXU/apple_silicon_nodes .
   ```

2. Install dependencies:
   ```bash
   pip install mlx numpy safetensors
   ```

3. Restart ComfyUI

## Apple Silicon Optimizations

### Memory Management
- **MLX lazy evaluation**: Computations are not executed until `mx.eval()` is called
- **Strategic eval points**: Only at bridge boundaries (PyTorch ↔ MLX)
- **Cache lifecycle**: `mx.clear_cache()` after sampling completes
- **Peak memory tracking**: Built into every node for debugging OOM

### Precision
- **float16 default**: Native on Apple Silicon, best performance
- **bfloat16 support**: Better numerical stability for sensitive ops
- **No torch.float16 on MPS**: Avoids NaN issues seen in PyTorch MPS

### Unified Memory
- **Zero-copy between CPU/GPU**: MLX and PyTorch MPS share the same physical memory
- **NumPy as bridge**: `mx.array(np_array)` creates a view, not a copy, when possible
- **Device placement**: Latents created on MPS for zero-copy with sampler

### Inference Speed
- **SeaCache**: Skip computation when previous step output is similar enough
- **TeaCache**: Output-level caching with adaptive threshold interpolation
- **Precomputed rope**: Positional embeddings computed once, reused across steps
- **Precomputed text projection**: `txt_in(emb)` computed once, reused across steps

### Advanced Conditioning
- **LoRA Runtime Loading**: Standard A/B matrices and ComfyUI diff format with per-LoRA alpha scaling
- **Multi-LoRA Stacking**: Up to 5 LoRAs applied simultaneously with strength scheduling
- **ControlNet Union**: 8 control types (pose, depth, soft_edge, line_canny, normal, segment, tile, repaint)
- **IP-Adapter**: Cross-attention injection using CLIP-Vision reference encoding
- **Kontext KV Cache**: Reference image tokens cached and injected into transformer attention layers

## Comparison with SDMLX

| Feature | SDMLX | ASDX |
|---------|-------|------|
| FLUX native transformer | ✅ (C++ core) | ✅ (pure MLX) |
| LoRA runtime loading | ✅ | ✅ |
| Multi-LoRA stacking | ✅ | ✅ |
| TeaCache acceleration | ✅ | ✅ |
| SeaCache acceleration | ✅ | ✅ |
| ControlNet Union | ✅ | ✅ |
| IP-Adapter | ✅ | ✅ |
| Kontext reference | ✅ | ✅ |
| Code size | ~15K lines | ~3K lines |
| Learning curve | Steep | Gentle |
| Modularity | Tightly coupled | Cleanly separated |

ASDX is the **minimal viable set** — the core diffusion pipeline with advanced features (LoRA, ControlNet, IP-Adapter, TeaCache) implemented in clean, readable MLX code. Use it as a starting point, learning reference, or production pipeline for Apple Silicon.

## License

MIT
