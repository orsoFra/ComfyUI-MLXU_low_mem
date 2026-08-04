"""ControlNet Union model for FLUX.1, and weight loading.

Matches the reference architecture in `comfy/ldm/flux/controlnet.py`
(`ControlNetFlux`): this is NOT a separate SDXL-style UNet — it's a
FLUX transformer that shares img_in/txt_in/time_in/double_blocks/
single_blocks with the base FLUX model, plus a `pos_embed_input`
projection for the (VAE-encoded) control latent and one linear
`controlnet_blocks[i]`/`controlnet_single_blocks[i]` projection per
block that produces the residual added into the base model's forward
pass (see FluxTransformer.__call__'s `control` parameter).

Contains:
  - ControlNetFlux: MLX-native ControlNet Union for FLUX
  - load_controlnet_union: load checkpoint and cache model
  - _assign_controlnet_weights: weight mapping utilities
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..native import (
    AXES_DIM,
    CONTEXT_IN_DIM,
    HIDDEN_DIM,
    ROPE_THETA,
    VEC_IN_DIM,
    DoubleBlock,
    MLPEmbedder,
    SingleBlock,
    embed_nd,
    timestep_embedding,
)
from ..native.config import FluxConfig


# ── ControlNet Union Model ───────────────────────────────────────────

class ControlNetFlux(nn.Module):
    """ControlNet Union for FLUX.1, matching comfy.ldm.flux.controlnet.ControlNetFlux.

    Architecture:
      pos_embed_input: Linear(control_latent_channels, hidden_dim) — projects
                        the VAE-encoded control latent, added to img tokens
      img_in / txt_in / time_in / vector_in / guidance_in: same as FluxTransformer
      double_blocks / single_blocks: same as FluxTransformer (config.num_*)
      controlnet_blocks[i]: Linear(hidden, hidden) per double block — produces
                             the residual injected into the base model's img
      controlnet_single_blocks[i]: same, per single block
      controlnet_mode_embedder: optional Embedding(num_union_modes, hidden) for
                                 ControlNet-Union's control-type conditioning
    """

    def __init__(
        self,
        config: FluxConfig | None = None,
        num_union_modes: int = 0,
        control_latent_channels: int = 16,
    ):
        super().__init__()
        config = config or FluxConfig()
        self.config = config
        self.dtype = config.mlx_dtype
        self.num_union_modes = num_union_modes

        self.img_in = nn.Linear(64, HIDDEN_DIM)
        self.txt_in = nn.Linear(CONTEXT_IN_DIM, HIDDEN_DIM)
        self.time_in = MLPEmbedder(256, HIDDEN_DIM)
        self.vector_in = MLPEmbedder(VEC_IN_DIM, HIDDEN_DIM)
        self.guidance_in = MLPEmbedder(256, HIDDEN_DIM) if config.guidance_embed else None

        # control_latent_channels is already *4 (2x2 patch) by the caller for
        # the packed-latent ControlNet-Union path (latent_input=True)
        self.pos_embed_input = nn.Linear(control_latent_channels, HIDDEN_DIM)

        self.double_blocks = [
            DoubleBlock() for _ in range(config.num_double_blocks)
        ]
        self.single_blocks = [
            SingleBlock() for _ in range(config.num_single_blocks)
        ]

        self.controlnet_blocks = [
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(config.num_double_blocks)
        ]
        self.controlnet_single_blocks = [
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(config.num_single_blocks)
        ]

        self.controlnet_mode_embedder = (
            nn.Embedding(num_union_modes, HIDDEN_DIM) if num_union_modes > 0 else None
        )

    def get_rope(self, img_h: int, img_w: int, txt_len: int) -> mx.array:
        """Same 3-axis RoPE table as FluxTransformer (shared position convention)."""
        img_len = img_h * img_w
        ids = mx.zeros((txt_len + img_len, 3), dtype=mx.float32)
        if img_len > 0:
            rows = mx.arange(img_h, dtype=mx.float32)[:, None]
            cols = mx.arange(img_w, dtype=mx.float32)[None, :]
            rows = mx.broadcast_to(rows, (img_h, img_w)).reshape(-1)
            cols = mx.broadcast_to(cols, (img_h, img_w)).reshape(-1)
            ids[txt_len:, 1] = rows
            ids[txt_len:, 2] = cols
        return embed_nd(ids, AXES_DIM, ROPE_THETA)

    def time_embed(self, t: mx.array, guidance: mx.array | None = None,
                   pooled: mx.array | None = None) -> mx.array:
        vec = self.time_in(timestep_embedding(t, 256).astype(self.dtype))
        if guidance is not None and self.guidance_in is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).astype(self.dtype))
        if pooled is not None:
            vec = vec + self.vector_in(pooled.astype(self.dtype))
        return vec

    def __call__(
        self,
        img: mx.array,               # [B, N_img, 64] packed noisy latent
        control_latent: mx.array,    # [B, N_img, control_latent_channels] packed control latent
        txt: mx.array,               # [B, N_txt, 4096] T5 embeddings
        t: mx.array,                 # [B] timestep
        img_h: int,
        img_w: int,
        guidance: mx.array | None = None,
        pooled: mx.array | None = None,
        rope: mx.array | None = None,
        control_type: list[int] | None = None,
    ) -> dict[str, list[mx.array]]:
        """Forward pass.

        Returns:
            {"input": [...N double residuals...], "output": [...N single residuals...]}
            matching FluxTransformer.__call__'s `control` parameter shape.
        """
        img = self.img_in(img.astype(self.dtype))
        img = img + self.pos_embed_input(control_latent.astype(self.dtype))
        txt = self.txt_in(txt.astype(self.dtype))

        vec = self.time_embed(t, guidance=guidance, pooled=pooled)

        if self.controlnet_mode_embedder is not None and control_type:
            mode_ids = mx.array(control_type)
            control_cond = self.controlnet_mode_embedder(mode_ids)[None]  # [1, M, hidden]
            control_cond = mx.broadcast_to(
                control_cond, (txt.shape[0], control_cond.shape[1], HIDDEN_DIM)
            )
            txt = mx.concatenate([control_cond, txt], axis=1)

        if rope is None:
            rope = self.get_rope(img_h, img_w, txt.shape[1])

        double_residuals = []
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, rope)
            double_residuals.append(img)
        double_out = [proj(r) for proj, r in zip(self.controlnet_blocks, double_residuals)]

        x = mx.concatenate([txt, img], axis=1)
        single_residuals = []
        for block in self.single_blocks:
            x = block(x, vec, rope)
            single_residuals.append(x[:, txt.shape[1]:])
        single_out = [proj(r) for proj, r in zip(self.controlnet_single_blocks, single_residuals)]

        return {"input": double_out, "output": single_out}


# ── Model Loader ─────────────────────────────────────────────────────

_CONTROLNET_CACHE: dict[str, ControlNetFlux] = {}


def load_controlnet_union(path: str | Path, dtype: str = "float16") -> ControlNetFlux:
    """Load a ControlNet Union model for FLUX from a checkpoint file."""
    from mlx.utils import tree_flatten, tree_unflatten

    path = Path(path)
    cache_key = f"{path}:{dtype}"
    if cache_key in _CONTROLNET_CACHE:
        return _CONTROLNET_CACHE[cache_key]

    from .. import native
    state = native._load_safetensors(path)

    # Strip common ComfyUI prefixes (matches native/weight_map.py::normalize_flux_keys)
    normalized: dict[str, mx.array] = {}
    for key, value in state.items():
        for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value

    # Same three renames as FLUX's map_flux_to_native: img_mlp./txt_mlp. -> _,
    # QKNorm scale -> weight. ControlNet checkpoints use the identical
    # double_blocks/single_blocks naming as the base FLUX model.
    mapped: dict[str, mx.array] = {}
    for key, value in normalized.items():
        new_key = key
        if ".img_mlp." in new_key:
            new_key = new_key.replace(".img_mlp.", ".img_mlp_")
        elif ".txt_mlp." in new_key:
            new_key = new_key.replace(".txt_mlp.", ".txt_mlp_")
        elif new_key.endswith(("query_norm.scale", "key_norm.scale")):
            new_key = new_key[: -len("scale")] + "weight"
        mapped[new_key] = value

    num_union_modes = 0
    if "controlnet_mode_embedder.weight" in mapped:
        num_union_modes = mapped["controlnet_mode_embedder.weight"].shape[0]

    control_latent_channels = 64  # default: same packed-latent width as img_in
    if "pos_embed_input.weight" in mapped:
        control_latent_channels = mapped["pos_embed_input.weight"].shape[1]

    config = FluxConfig(dtype=dtype)
    model = ControlNetFlux(
        config, num_union_modes=num_union_modes,
        control_latent_channels=control_latent_channels,
    )

    model_flat = tree_flatten(model.parameters())
    new_flat = []
    matched = 0
    for flat_key, value in model_flat:
        if flat_key in mapped:
            new_flat.append((flat_key, mapped[flat_key]))
            matched += 1
        else:
            new_flat.append((flat_key, value))
    model.update(tree_unflatten(new_flat))
    mx.eval(model.parameters())

    print(f"[ASDX] ControlNet Union loaded: {path.name} "
          f"({matched}/{len(model_flat)} params matched)")
    _CONTROLNET_CACHE[cache_key] = model
    return model
