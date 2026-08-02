# Plan : Multi-modeles Apple Silicon

**Date de creation :** 2026-08-02
**Status :** Planification — reference architecturale
**Objectif :** Transformer le projet en plateforme multi-modeles optimisee Apple Silicon, au-dela de FLUX.1.

---

## 1. Vision

Passer de **FLUX.1-only** a **tous les modeles de diffusion** sur infrastructure MLX native.

### Principes directeurs

1. **Interface commune** — chaque famille de modeles implemente les memes interfaces (transformer, VAE, text encoder)
2. **Dispatch automatique** — le loader detecte le type de modele depuis le checkpoint
3. **Zero copie superflue** — tenseurs en memoire unifiee, bridge MLX↔PyTorch minimal
4. **Retrocompatibilite** — workflows existants continuent de fonctionner
5. **Extensibilite** — ajouter un modele = ajouter un module, pas modifier le noyau

---

## 2. Modeles cibles

| Famille | Modeles | VAE | Latent | Encoder | Scheduler | Priorite |
|---------|---------|-----|--------|---------|-----------|----------|
| **FLUX.1** | dev, schnell, fill, depth | 16 can. | 16 | T5-XXL + CLIP-L | Euler | ✅ Deja supporte |
| **FLUX.2** | Klein | 16 can. | 16 | T5-XXL + CLIP-L | A definir | ⭐ Haute |
| **SD1.5** | v1.5, v1.4, v2.1 | 4 can. | 4 | CLIP-L | DDIM, Euler, DPM++ | ⭐ Haute |
| **SDXL** | base, refiner | 4 can. | 4 | CLIP-L + OpenCLIP | DDIM, Euler, DPM++ | ⭐ Haute |
| **Wan 2.1** | 480p, 720p, 14B | 4 can. | 4 | UMT5-XXL | Euler | Moyenne |
| **Hunyuan** | DiT XL/2 | 4 can. | 4 | CLIP + BERT | DDIM | Moyenne |
| **PixArt** | Sigma 2B | 4 can. | 4 | T5-XXL + CLIP-L | Euler | Basse |
| **Krea2** | Base, Turbo | ? | ? | Krea2 CLIP | A definir | Basse |
| **SVD** | SVD, SV3D | 4 can. | 4 | CLIP-L | Euler | Basse |

---

## 3. Architecture cible

```
apple_silicon_nodes/
├── __init__.py                    # Registration unifiee
├── bridge.py                      # PyTorch ↔ MLX (existe, a etendre)
├── capability.py                  # Profiles (existe, a etendre)
│
├── loader.py                      # Dispatcher unifie (a refondre)
│
├── conditioning.py                # CLIP conditioning (existe, a etendre)
│
├── sampler/
│   ├── __init__.py                # Node MLX Sampler
│   ├── core.py                    # Boucle denoising (a refondre)
│   ├── cache.py                   # TeaCache, Kontext (existe)
│   └── scheduler.py               # NOUVEAU — schedulers par modele
│
├── native/
│   ├── __init__.py                # Export commun
│   ├── config.py                  # FluxConfig (existe → TransformerConfig)
│   ├── transformer/               # NOUVEAU — transformers par famille
│   │   ├── base.py                # Interface DiffusionTransformer
│   │   ├── flux.py                # FLUX.1 (deplace de native/__init__.py)
│   │   ├── flux2.py               # FLUX.2-Klein
│   │   ├── sd_unet.py             # SD1.5/SDXL UNet
│   │   ├── hunyuan_dit.py         # Hunyuan DiT
│   │   ├── wan_video.py           # Wan 2.1
│   │   ├── pixart_sigma.py        # PixArt Sigma
│   │   ├── krea2.py               # Krea2
│   │   └── svd.py                 # Stable Video Diffusion
│   ├── vae/                       # NOUVEAU — VAE par famille
│   │   ├── base.py                # Interface DiffusionVAE
│   │   ├── flux_vae.py            # FLUX VAE (deplace de mlx_vae.py)
│   │   ├── sd_vae.py              # SD VAE 4 canaux
│   │   ├── wan_vae.py             # Wan VAE
│   │   └── hunyuan_vae.py         # Hunyuan VAE
│   ├── text_encoder/              # NOUVEAU — encodeurs de texte
│   │   ├── base.py                # Interface TextEncoder
│   │   ├── t5_encoder.py          # T5-XXL MLX
│   │   ├── clip_encoder.py        # CLIP-L/G MLX
│   │   ├── umt5_encoder.py        # UMT5-XXL (Wan)
│   │   ├── bert_encoder.py        # BERT-Large (Hunyuan)
│   │   └── qwen_encoder.py        # Qwen (Qwen Image)
│   └── rope/                      # NOUVEAU — embeddings positionnels
│       ├── base.py                # Interface RoPE
│       ├── flux_rope.py           # FLUX 2D spatial rope
│       ├── sd_rope.py             # SD 1D sinusoidal
│       └── wan_rope.py            # Wan 3D temporal rope
│
├── lora.py                        # LoRA (a etendre)
├── controlnet/                    # ControlNet (a etendre)
│   ├── __init__.py
│   ├── types.py
│   ├── blocks.py
│   ├── model.py
│   └── sd_controlnet.py           # NOUVEAU — ControlNet SD
├── ip_adapter.py                  # IP-Adapter (assez generique)
├── image_chain.py                 # Image chain (generique)
├── depth_map.py                   # Depth map (generique)
├── live_preview.py                # Live preview (a adapter)
├── memory.py                      # Memory profiler (generique)
├── latent.py                      # Empty latent (a etendre)
└── vae.py                         # VAE nodes (a etendre)
```

---

## 4. Interfaces cibles

### 4.1 DiffusionTransformer (interface generique)

```python
# apple_silicon_nodes/native/transformer/base.py

@dataclass(frozen=True)
class TransformerConfig:
    """Configuration generique pour tous les transformers."""
    family: str                      # "flux1", "sd1", "sdxl", "hunyuan", "wan"
    name: str                        # "FLUX.1-dev", "SD1.5", etc.
    latent_channels: int             # 16 (FLUX) ou 4 (SD/Wan/Hunyuan)
    hidden_dim: int
    num_heads: int
    dtype: str = "float16"
    supports_guidance: bool = True
    supports_img2img: bool = False
    supports_inpainting: bool = False
    supports_video: bool = False
    supports_controlnet: bool = False


class DiffusionTransformer(ABC):
    """Interface generique pour tous les transformers de diffusion."""
    
    @abstractmethod
    def predict(
        self,
        latent: mx.array,     # [B, C_latent, H_lat, W_lat]
        t: mx.array,          # [B] timestep
        cond: mx.array,       # [B, T, D] conditioning texte
        cond_pooled: mx.array | None,  # [B, D_pooled] pooling
        guidance: float | None = None,
        **kwargs,
    ) -> mx.array:
        """Predict noise/residual."""
        ...
    
    @abstractmethod
    def get_config(self) -> TransformerConfig: ...
    @abstractmethod
    def get_latent_channels(self) -> int: ...
    @abstractmethod
    def unpack_latent(self, packed: mx.array, h: int, w: int) -> mx.array: ...
    @abstractmethod
    def pack_latent(self, latent: mx.array) -> mx.array: ...
```

### 4.2 DiffusionVAE (interface generique)

```python
# apple_silicon_nodes/native/vae/base.py

class DiffusionVAE(ABC):
    """Interface generique pour tous les VAE."""
    
    @abstractmethod
    def encode(self, image: mx.array) -> mx.array:
        """Encode image [B, 3, H, W] → latent [B, C, H/8, W/8]."""
        ...
    
    @abstractmethod
    def decode(self, latent: mx.array) -> mx.array:
        """Decode latent [B, C, H, W] → image [B, 3, H, W]."""
        ...
    
    @abstractmethod
    def get_latent_channels(self) -> int: ...
```

### 4.3 TextEncoder (interface generique)

```python
# apple_silicon_nodes/native/text_encoder/base.py

class TextEncoder(ABC):
    """Interface generique pour les encodeurs de texte."""
    
    @abstractmethod
    def tokenize(self, texts: list[str], max_tokens: int = 256) -> dict: ...
    @abstractmethod
    def encode(self, tokens: dict) -> tuple[mx.array, mx.array | None]:
        """Returns (embeddings [B,T,D], pooled [B,D_pooled] or None)."""
        ...
    @abstractmethod
    def get_embedding_dim(self) -> int: ...
    @abstractmethod
    def is_pooled(self) -> bool: ...
```

### 4.4 Scheduler (interface generique)

```python
# apple_silicon_nodes/sampler/scheduler.py

class DiffusionScheduler(ABC):
    """Interface generique pour les schedulers."""
    
    @abstractmethod
    def set_timesteps(self, num_steps: int, model_type: str) -> None: ...
    @abstractmethod
    def timesteps(self) -> list[float]: ...
    @abstractmethod
    def step(
        self,
        model_output: mx.array,
        timestep: float,
        sample: mx.array,
        **kwargs,
    ) -> mx.array:
        """Compute previous sample."""
        ...
```

---

## 5. Phases d'implementation

### Phase 0 — Fondations (refactoring)

**Objectif :** Creer les interfaces abstraites sans casser le code existant.

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| `native/transformer/base.py` | Interface DiffusionTransformer | 80 | 2h |
| `native/vae/base.py` | Interface DiffusionVAE | 30 | 30min |
| `native/text_encoder/base.py` | Interface TextEncoder | 40 | 30min |
| `sampler/scheduler.py` | Schedulers generiques | 200 | 3h |
| `loader.py` | Dispatcher unifie | ~150 | 1j |
| `sampler/core.py` | Abstraction transformer | ~100 | 1j |
| `capability.py` | Profils nouveaux modeles | ~100 | 3h |
| `bridge.py` | Support multi-canaux | ~50 | 2h |
| `native/config.py` | Renommer → TransformerConfig | 20 | 30min |
| `native/__init__.py` | Re-export transformers | 30 | 30min |
| `__init__.py` | Registration nouveaux nodes | 20 | 1h |

**Total phase 0 : 5-6 jours**

---

### Phase 1 — SD1.5 / SDXL (priorite haute)

**Objectif :** Supporter SD1.5 et SDXL — demande la plus courante apres FLUX.

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| `native/transformer/sd_unet.py` | SD1.5/SDXL UNet MLX | 700 | 2-3j |
| `native/vae/sd_vae.py` | SD VAE 4 canaux MLX | 250 | 1j |
| `controlnet/sd_controlnet.py` | ControlNet SD MLX | 400 | 2-3j |
| `lora.py` | Mapping cles LoRA SD | 50 | 2h |
| `latent.py` | Support multi-canaux | 30 | 1h |
| `vae.py` | Dispatch VAE | 50 | 2h |
| `conditioning.py` | Support multi-encoder | 50 | 2h |

**Total phase 1 : 8-11 jours**

---

### Phase 2 — FLUX.2 Klein (priorite haute)

**Objectif :** Supporter FLUX.2-Klein — evolution directe de FLUX.1.

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| `native/transformer/flux2.py` | FLUX.2-Klein MLX | 500 | 2-3j |

**Total phase 2 : 2-3 jours**

---

### Phase 3 — Wan 2.1 (priorite moyenne)

**Objectif :** Supporter Wan 2.1 — modele video.

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| `native/transformer/wan_video.py` | Wan 2.1 video MLX | 1000 | 3-5j |
| `native/vae/wan_vae.py` | Wan VAE MLX | 250 | 1-2j |
| `native/text_encoder/umt5_encoder.py` | UMT5-XXL MLX | 300 | 1-2j |

**Total phase 3 : 5-8 jours**

---

### Phase 4 — Hunyuan (priorite moyenne)

**Objectif :** Supporter Hunyuan DiT.

| Fichier | Contenu | Lignes | Effort |
|---------|---------|--------|--------|
| `native/transformer/hunyuan_dit.py` | Hunyuan DiT MLX | 600 | 2-3j |
| `native/vae/hunyuan_vae.py` | Hunyuan VAE MLX | 250 | 1-2j |
| `native/text_encoder/bert_encoder.py` | BERT-Large MLX | 250 | 1-2j |

**Total phase 4 : 4-6 jours**

---

### Phase 5 — Modeles avances (priorite basse)

| Fichier | Modele | Lignes | Effort |
|---------|--------|--------|--------|
| `native/transformer/pixart_sigma.py` | PixArt Sigma | 400 | 2j |
| `native/transformer/krea2.py` | Krea2 | 400 | 2j |
| `native/transformer/ideogram.py` | Ideogram | 400 | 2j |
| `native/transformer/svd.py` | SVD | 500 | 2-3j |

**Total phase 5 : 8-10 jours**

---

## 6. Resume des efforts

| Phase | Fichiers nouveaux | Fichiers modifies | Effort |
|-------|-------------------|-------------------|--------|
| **0 — Fondations** | 5 | 8 | **5-6j** |
| **1 — SD1.5/SDXL** | 3 | 5 | **8-11j** |
| **2 — FLUX.2** | 1 | 0 | **2-3j** |
| **3 — Wan 2.1** | 3 | 0 | **5-8j** |
| **4 — Hunyuan** | 3 | 0 | **4-6j** |
| **5 — Autres** | 4 | 0 | **8-10j** |
| **TOTAL** | **22** | **13** | **~32-46j** |

**Avec 2 developpeurs :** ~16-23 jours

---

## 7. Memoire unifiee — strategie

### Regles d'or

1. **Un seul buffer par tenseur** — jamais de copie inutile entre MLX et PyTorch
2. **Eval strategique** — `mx.eval()` uniquement aux points de bridge
3. **Cache MLX limite** — `mx.set_cache_limit()` adapte a la RAM disponible
4. **Nettoyage inter-phases** — `mx.clear_cache()` entre loader, encode, sampling, decode

### Conversion MLX ↔ PyTorch (bridge)

```
PyTorch → MLX (input):
  tensor.detach().cpu().numpy() → mx.array() → mx.eval()

MLX → PyTorch (output):
  mx.eval() → np.array() → torch.from_numpy().to("mps")

Regles:
  - Minimiser les conversions, batcher quand possible
  - Toujours appeler mx.eval() avant conversion MLX→NumPy
  - Utiliser des buffers numpy reutilisables quand possible
```

### Memoire par modele (FP16)

| Modele | RAM min. recommandee | Poids + Etat | Avec TeaCache |
|--------|---------------------|--------------|---------------|
| SD1.5 | 8GB | ~3GB | ~2GB |
| SDXL | 16GB | ~5GB | ~3GB |
| FLUX.1 | 16GB | ~8GB | ~5GB |
| FLUX.2 | 24GB | ~12GB | ~7GB |
| Wan 2.1 | 36GB | ~18GB | ~10GB |
| Hunyuan | 16GB | ~6GB | ~4GB |

---

## 8. Tests de validation

### Tests numeriques (par modele)

- [ ] Predictions stables (pas de NaN/Inf)
- [ ] Similarite cosinus > 0.99 vs PyTorch reference
- [ ] Embeddings textuels coherents

### Tests fonctionnels (par modele)

- [ ] Workflow complet : charge → encode → sample → decode → image
- [ ] img2img fonctionne
- [ ] inpainting fonctionne (si supporte)
- [ ] LoRA fonctionne
- [ ] ControlNet fonctionne (si supporte)
- [ ] IP-Adapter fonctionne (si supporte)

### Tests de performance (par modele)

- [ ] Temps de generation mesure et documente
- [ ] Memoire maximale mesuree
- [ ] Comparaison avec implementation PyTorch reference

### Tests de compatibilite (par modele)

- [ ] Resolutions supportees documentees
- [ ] Batch size 1 garanti
- [ ] Float16 et bfloat16 testes
- [ ] Fallback CPU teste

---

## 9. Ordre d'execution

```
Phase 0 (fondations) ───────────────────────────────────── 5-6j
    │
    ├──→ Phase 1 (SD1.5/SDXL) ──────────────────────────── 8-11j
    │       │
    │       ├──→ Tests SD
    │       └──→ Documentation SD
    │
    ├──→ Phase 2 (FLUX.2) ───────────────────────────────── 2-3j
    │       │
    │       └──→ Tests FLUX.2
    │
    ├──→ Phase 3 (Wan 2.1) ──────────────────────────────── 5-8j
    │       │
    │       └──→ Tests Wan
    │
    ├──→ Phase 4 (Hunyuan) ──────────────────────────────── 4-6j
    │       │
    │       └──→ Tests Hunyuan
    │
    └──→ Phase 5 (Autres) ───────────────────────────────── 8-10j
            │
            └──→ Tests PixArt, Krea2, Ideogram, SVD
```

---

## 10. Risques et mitigations

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| Architecture modele inconnue (pas de specs) | Moyenne | Eleve | Analyser les poids pour deduire l'architecture |
| Memoire insuffisante (Wan 14B) | Moyenne | Eleve | Quantification MLX (int8, fp8), streaming |
| Incompatibilite numerique MLX vs PyTorch | Haute | Moyen | Tests de validation rigoureux |
| Schedulers complexes (DPM++ SDE) | Moyenne | Moyen | Utiliser schedulers ComfyUI via bridge |
| Temps > estime | Haute | Eleve | Prioriser SD1.5/SDXL (demande la plus forte) |
| ControlNet non disponible pour certains modeles | Basse | Moyen | ControlNet Union couvre FLUX/SD |

---

## 11. Migration progressive

### Backward compatibility

```python
# loader.py — compatibilite arriere

def load_diffusion_model(path: str | Path, dtype: str = "float16"):
    """Charge le bon transformer selon le modele detecte.
    
    Retourne un objet implementant DiffusionTransformer.
    Pour FLUX.1, retourne FluxTransformer (compatible arriere).
    """
    family = _detect_model_family(path)
    
    if family == "flux1":
        return _load_existing_flux(path, dtype)  # code existant
    
    loader = _get_loader(family)
    return loader(path, dtype)
```

### Flag de commutation

```python
# Variable d'environnement pour activer le dispatcher multi-modeles
_USE_DISPATCHER = os.environ.get("ASDX_MULTI_MODEL", "0") == "1"
# Par defaut: comportement actuel (FLUX.1 only)
```

---

## 12. Checklist de livraison

### Phase 0
- [ ] Interfaces creees et testeess
- [ ] Dispatcher fonctionne pour FLUX.1 (backward compat)
- [ ] Tests unitaires passes

### Phase 1 (SD1.5/SDXL)
- [ ] UNet SD charge et genere sans erreur
- [ ] VAE SD encode/decode correctement
- [ ] Schedulers SD testes (Euler, Euler-A, DDIM, DPM++2M)
- [ ] LoRA SD fonctionne
- [ ] ControlNet SD fonctionne
- [ ] Workflow complet teste
- [ ] Memoire et performances mesurees

### Phase 2-5
- [ ] Chaque modele passe les memes tests que SD
- [ ] Documentation a jour

### General
- [ ] Tous les tests passes
- [ ] Benchmarks compares avec PyTorch
- [ ] Graphify run pour verifier couverture

---

## 13. References

- FLUX.1: https://github.com/black-forest-labs/flux
- SD1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5
- SDXL: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- FLUX.2: https://github.com/black-forest-labs/flux
- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- Hunyuan DiT: https://github.com/Tencent/HunyuanDiT
- PixArt Sigma: https://github.com/PixArt-alpha/PixArt-sigma
- MLX: https://ml-explore.github.io/mlx/
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
