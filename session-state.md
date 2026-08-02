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

---

## Session 3 — Unified CLIP Text Encode

**Date:** 2026-08-02 (continuation)

### What was done

#### Unification de `ASDX_CLIPTextEncodeFlux` + `ASDX_CLIPTextEncode`
- **Problème :** Deux noeuds séparés pour l'encodage texte (FLUX vs SD), ce qui obligeait l'utilisateur à choisir le bon noeud manuellement.
- **Fix :** Un seul noeud `ASDX_CLIPTextEncode` qui détecte automatiquement le mode :
  - Si `t5xxl` est fourni → mode FLUX (encode séparé clip_l + t5xxl + guidance)
  - Si `t5xxl` est vide → mode SD/SDXL/Pony (encode CLIP standard)
- Réduction de -30 lignes (fusion de 2 classes en 1)
- Suppression de `ASDX_CLIPTextEncodeFlux` (obsolète)
- Mises à jour : docstring, `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, `__init__.py`

#### Commit
- `e835a61` — "feat(conditioning): unify CLIP Text Encode (FLUX + SD/SDXL/Pony in one node)"

### Key decisions
- Détection automatique via paramètre optionnel `t5xxl` (heuristic simple et robuste)
- Pas de dropdown "mode" explicite — le CLIP est déjà chargé avec le bon type
- `t5xxl` en paramètre optionnel avec `default=""` — ComfyUI passe `""` si non connecté

---

## Session 2 — Fix Checkpoint/Diffusion Loader + Graphify

**Date:** 2026-08-02 (continuation)

### What was done

#### Fix: ASDX_DiffusionLoader (broken placeholder → real loader)
- **Problème:** Le noeud `ASDX_DiffusionLoader` créait un `FluxTransformer(config)` vide sans charger aucun poids depuis le checkpoint.
- **Fix:** Appel de `native.load_transformer(path, dtype=precision)` qui charge les poids safetensors dans le transformer.
- **Commit:** `fecfb9d` — "fix(loader): implement real Diffusion Loader + add Checkpoint Loader"

#### Ajout: ASDX_CheckpointLoader
- Nouveau noeud qui charge le checkpoint complet (VAE + CLIP + Diffusion)
- Retourne `("asdx_model", "mlx_clip", "mlx_vae")` — pattern SDMLX
- Placeholders pour CLIP et VAE (gérés par DualCLIPLoader et nodes VAE séparés)
- Découverte automatique des checkpoints dans `checkpoints/` et `diffusion_models/`

#### Distinction Checkpoint Loader vs Diffusion Loader
- **Diffusion Loader:** charge uniquement le transformer (UNet) depuis un fichier diffusion model
- **Checkpoint Loader:** charge le checkpoint complet (transformer + placeholders CLIP + VAE)
- Inspiré de `SDMLX_LoaderUniversal` (full) vs `SDMLX_Loader` (packages)

#### Graphify — Knowledge Graph du projet
- Pipeline graphify exécuté sur le répertoire complet
- **Résultats:** 426 nœuds, 649 arêtes, 18 communautés
- **Output:** `graphify-out/graph.html` (202 Ko), `graphify-out/graph.json` (381 Ko), `graphify-out/GRAPH_REPORT.md`
- **FluxConfig** identifié comme le nœud le plus connecté du graphe — pont entre 7 communautés
- Analyse complète des 130+ arêtes rayonnant depuis FluxConfig

### Key decisions
- FluxConfig est un `@dataclass(frozen=True)` — abstraction légitime, pas un anti-pattern de couplage
- FluxConfig connecte 7 des 18 communautés (Model Loader, Dtype, Transformer Blocks, FluxTransformer Core, Config, RoPE, Sampler)
- 14 nœuds isolés identifiés (T5-XXL, LoRA Schedule, Diffusion sampler Euler) — possibles gaps de documentation

### Fichiers modifiés
- `loader.py` — fix Diffusion Loader + nouveau Checkpoint Loader
- `__init__.py` — ajout du display name pour Checkpoint Loader

### Fichiers générés
- `graphify-out/graph.html` — visualisation interactive
- `graphify-out/graph.json` — données brutes du graphe
- `graphify-out/GRAPH_REPORT.md` — rapport d'audit
- `graphify-out/cost.json` — suivi des coûts
- `graphify-out/.graphify_manifest.json` — manifest d'extraction
