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

## Session 5 — Revue LoRA, comparaison inter-projets, performance et mémoire Krea2

**Date:** 2026-08-06

### What was done

#### Revue de code des nœuds LoRA (`/code-review` + `/simplify`)
Audit ligne à ligne de `apple_silicon_nodes/lora.py` demandé explicitement par
l'utilisateur. Deux bugs de parsing confirmés et corrigés :
- **`.lora_up.`/`.lora_down.` (kohya-style)** : la recherche de clé était un no-op
  (`up_key`/`down_key` dérivés de `prefix` déjà nettoyé, donc toujours identiques et
  jamais trouvés dans `raw`) — ce format était silencieusement ignoré à 100%.
- **`.diff`/`.diff_b` (format diff ComfyUI)** : la clé cible reconstruite omettait le
  suffixe réel `.weight`/`.bias` (vérifié contre le vrai `comfy/lora.py` : la clé source
  ne le contient jamais) — navigation vers le mauvais niveau du module, no-op silencieux
  ou crash.
Les deux corrigés en routant à travers le même pipeline `deltas`/conversion B@A que le
format standard `.lora_A.`/`.lora_B.` (jamais touché, toujours correct).

#### Comparaison avec 5 projets frères (Sampler + LoRA)
À la demande de l'utilisateur, exploration parallèle de tous les autres dépôts de
`/Volumes/MBP2021/Images/Projet/` (ComfyUI-mflux-AnyModel, comfyui-mlx, Mflux-ComfyUI,
SDMLX, SceneWorks) pour comparer notre traitement du Sampler/LoRA. Trois constats
convergents, confirmés par 3-4 projets indépendants et directement actionnables :
1. Tous utilisent `mx.fast.scaled_dot_product_attention` (kernel Metal fusionné) au lieu
   d'un matmul manuel — nous étions les seuls à ne pas le faire.
2. Le cache modèle de tous les projets comparés inclut l'identité du LoRA (path/scale/
   fingerprint) dans sa clé — nous étions les seuls à l'ignorer.
3. Le conditioning texte est calculé une fois hors boucle partout ailleurs — nous le
   recalculions à chaque step.
mflux et SDMLX appliquent aussi le LoRA de façon **non-destructive** (wrapper `LoRALinear`
au forward, jamais de mutation des poids de base) pour les formats standard/kohya.

#### Le vrai bug de correction : cache LoRA destructif + compoundage
Test utilisateur (même checkpoint/LoRA/CFG/steps/seed, notre pipeline vs référence) a
révélé deux problèmes réels : images différentes malgré paramètres identiques, et 19 min
vs 4 min de temps de génération. Investigation multi-étapes (fausses pistes explorées et
écartées avec preuves : throttling thermique, pression mémoire générique, cumul LoRA sur
la toute première comparaison) a abouti à la vraie cause :
`_apply_lora_to_transformer` **mutait en place** le transformer partagé par
`loader.py::_MODEL_CACHE` (jamais purgé pour une même clé) — un LoRA déjà appliqué lors
d'un run précédent se voyait ré-appliqué par-dessus lui-même si le nœud LoRA se
ré-exécutait dans la même session ComfyUI, cumulant l'effet.

**Fix** : `_apply_lora_to_transformer` retourne maintenant un **nouveau** transformer
(via `mlx.utils.tree_flatten`/`tree_unflatten`, poids non modifiés partagés par
référence, seuls les deltas LoRA recalculés) — le modèle en cache n'est plus jamais
muté. Compromis assumé : pic mémoire plus élevé (base + copie LoRA'ée coexistent le temps
d'une génération).

#### Audit mémoire sur l'ensemble du projet
Ce compromis mémoire a provoqué un plantage silencieux lors d'un second test LoRA dans la
même session (sans redémarrage ComfyUI). Trois audits parallèles (loaders/LoRA/
ControlNet/IP-Adapter, runtime sampler, VAE/CLIP/bridge) ont couvert tout le dépôt.
Corrections appliquées :
- `lora.py` : `bridge.clear_mlx_cache()` ajouté après chaque application LoRA (les 3
  sites) — cause la plus probable du plantage, les buffers orphelins de l'ancien
  transformer LoRA'é n'étaient jamais réclamés par l'allocateur MLX.
- Même pattern d'éviction "une seule entrée résidente" que `_MODEL_CACHE` répliqué sur
  `_CONTROLNET_CACHE`, `_CLIP_CACHE`, `_DEPTH_CACHE` (jamais purgés auparavant).
- `low_memory_mode` (5 blocs dupliqués dans `sampler/core.py`, un par famille de modèle) :
  triplement inopérant (référence partagée avec `_MODEL_CACHE`, `_SamplerCore` de toute
  façon local, `mx.clear_cache()` en doublon) — corrigé en évinçant réellement
  `loader.py::_MODEL_CACHE` via une nouvelle fonction `clear_model_cache()`.
- `LivePreviewRegistry` : chaque génération ajoutait un callback permanent (nouvelle
  closure à chaque fois, le garde anti-doublon ne se déclenchait jamais) — purge en
  début de `register()`.

#### Performance Krea2 : ~88s/step → ~5-9s/step
Deux fixes distincts, tous deux confirmés par la comparaison inter-projets :
- **Attention fusionnée GPU** : `SingleStreamBlock` (les 28 blocs principaux, séquence
  image complète ~4444 tokens) forçait ses matmuls d'attention sur le stream CPU de MLX
  — un fix de précision hérité de l'investigation "piqueté" (Session 4), valide pour
  `TextFusionBlock` (séquence texte courte) mais catastrophique en vitesse pour la
  séquence image complète. Remplacé par `mx.fast.scaled_dot_product_attention` sur GPU
  (GQA natif, softmax interne en float32). Vérifié numériquement identique au bit près
  au chemin manuel GPU. Aucune réapparition du piqueté après le changement.
- **Conditioning texte hors boucle** : `SingleStreamDiT.encode_text()` (txtfusion +
  enhancer Krea2T) calculé une fois avant la boucle de sampling au lieu d'être recalculé
  à chaque step (jusqu'à 12s/step perdus).

#### Le vrai bug de correction restant : alpha LoRA jamais lu du fichier
Après tous les fixes ci-dessus, différence de visage persistante (référence : visage
demandé clairement reconnaissable ; nous : ressemblance mais pas le bon visage) à
strength 1.0 comme 1.5. Root cause : `_load_lora_file` codait `alpha=1.0` en dur, sans
jamais lire la vraie clé `"<prefix>.alpha"` du fichier LoRA (convention standard, lue
par `comfy/lora.py:41-45`). Pire : quand le fichier n'a pas de clé alpha du tout, le
vrai ComfyUI applique un scale de **1.0 complet** (pas de division par rank), alors que
notre code divisait systématiquement par `rank` — sous-application par un facteur égal
au rank (32x pour ce LoRA). Confirmé par le log : `scale=0.0312 ≈ 1/32` avant le fix,
`scale=1.0000` après (alpha du fichier égal au rank, convention d'entraînement courante).

**Fix** : nouvelle fonction partagée `base_lora_scale(alpha, rank)` reproduisant
exactement le fallback de comfy (`alpha/rank` si alpha présent, sinon `1.0` plat).
Lecture de la clé `.alpha` ajoutée à `_load_lora_file`. Widget `"alpha"` de
`ASDX_LoraLoader` **supprimé** (le vrai ComfyUI `LoraLoader` n'en a pas — alpha vient
toujours du fichier).

#### Commit
- `0e785ce` — "fix(krea2): correct LoRA scaling/caching bugs, fuse attention, cut Krea2 sampling time ~9x"

### Key decisions
- LoRA appliqué de façon non-destructive (nouvelle instance de transformer par
  application) plutôt que de synchroniser la clé de cache entre le loader et les nœuds
  LoRA — élimine structurellement le risque de compoundage, au prix d'un pic mémoire
  ~2x pendant une génération avec LoRA (acceptable sur machine 64Go+, à surveiller sur
  machines à mémoire plus limitée).
- Widget "alpha" manuel supprimé plutôt que gardé comme override — aligné sur le
  comportement natif ComfyUI, alpha vient toujours du fichier LoRA.
- Attention CPU-stream gardée uniquement pour `TextFusionBlock` (séquence courte) ; la
  précision qu'elle apporte n'était de toute façon pas la cause du "piqueté" (Wan21,
  Session 4) et son coût était disproportionné sur la séquence image complète.
- Méthodologie validée une seconde fois : comparer l'implémentation avec plusieurs
  projets frères indépendants (pas seulement le code source ComfyUI de référence) fait
  émerger des patterns de conception (cache-key LoRA, attention fusionnée, non-
  destructivité) qu'une simple relecture du code local n'aurait pas signalés.

### Fichiers modifiés
- `apple_silicon_nodes/lora.py` — parsing kohya/diff, application non-destructive,
  `bridge.clear_mlx_cache()`, lecture du vrai alpha, suppression du widget alpha
- `apple_silicon_nodes/native/krea2/model.py` — attention fusionnée GPU pour
  `SingleStreamBlock`, `encode_text()` hoisté hors boucle
- `apple_silicon_nodes/sampler/core.py` — appels LoRA schedule mis à jour, fix
  `low_memory_mode` (5 blocs), import `gc` retiré (devenu inutile)
- `apple_silicon_nodes/loader.py` — nouvelle fonction `clear_model_cache()`
- `apple_silicon_nodes/controlnet/model.py`, `conditioning.py`, `depth_map.py` —
  éviction des caches `_CONTROLNET_CACHE`/`_CLIP_CACHE`/`_DEPTH_CACHE`
- `apple_silicon_nodes/live_preview.py` — purge du registre avant chaque enregistrement

---

## Session 4 — Krea2 "piqueté" investigation and fix

**Date:** 2026-08-05

### What was done

#### Le bug : texture en croisillon ("piqueté") sur toutes les générations Krea2
Investigation très longue et méthodique d'un défaut visuel systématique (motif fin en
damier) sur les images Krea2 générées par ASDX, absent des générations produites par un
workflow ComfyUI natif de référence utilisant le même checkpoint/prompt/seed.

**Pistes explorées et corrigées en chemin (réelles, mais pas la cause principale) :**
- Précision Metal du GPU sur les matmuls d'attention (routées vers le stream CPU de MLX)
- Préservation de la précision F32 native du checkpoint pour les couches limites
- Formule et grille de timesteps du schedule sigma (utilisait `time_snr_shift`, faux pour
  Krea2 qui s'enregistre comme `ModelSamplingFlux` côté Comfy — corrigé vers
  `flux_time_shift` + grille linéaire en espace timestep, vérifié bit-exact contre
  `comfy.samplers.normal_scheduler`)
- Port de l'enhancer communautaire `ComfyUI-Krea2T-Enhancer` (boost du conditioning texte)

**Root cause réelle, trouvée via un audit multi-agents avec regard neuf** (comfy-reference-diff
+ weight-map-reviewer + 2 agents généralistes en parallèle) : `comfy.latent_formats.Wan21`
(l'espace latent de Krea2) définit une vraie transformation affine par canal
(`process_in`/`process_out`, 16 constantes moyenne/écart-type), qu'on croyait à tort être une
identité (`scale_factor=1.0` n'annule qu'une partie de la formule). Cette transformation
n'était jamais appliquée en sortie du sampler — d'où un latent final à ~moitié de l'amplitude
réelle, ce qui déclenche l'artefact via le décodeur WanVAE (très sensible à l'amplitude
d'entrée).

**Vérification numérique** : après le fix, le ratio d'amplitude par canal (qui allait de
1.10× à 3.16× hors cible) est tombé à 0.97-1.02× sur les 16 canaux, comparé à un latent de
référence Comfy natif au même seed.

#### Commits
- `bcc1cf1` — "fix(krea2): apply Wan21's real per-channel latent de-whitening"
- `520a8bd` — "chore(graphify): rebuild knowledge graph after Krea2 fixes"

### Key decisions
- L'enhancer Krea2T est gardé (paramètre `krea2_enhancer_strength`, défaut 1.0) car le
  workflow de référence l'utilise réellement — mais ce n'est PAS le fix de l'amplitude,
  juste une fonctionnalité légitime à part (effet mesuré ~2-4% sur la sortie finale).
- Le rescale empirique temporaire (`krea2_output_rescale=1.95`) a été retiré, remplacé par
  la vraie transformation par canal.
- `native/krea2/config.py::KREA2_LATENT_SCALE/SHIFT` (mauvaises constantes FLUX, code mort)
  supprimées.
- Méthodologie validée : quand une investigation approfondie piétine malgré de multiples
  vérifications individuelles cohérentes, un audit multi-agents en parallèle avec instruction
  explicite de ne pas faire confiance aux conclusions précédentes permet de repérer une
  hypothèse fondamentale erronée (ici : "Wan21 = identité") qui avait biaisé toute
  l'investigation précédente.

### Fichiers modifiés
- `apple_silicon_nodes/native/config.py` — `WAN21_LATENTS_MEAN/STD`, `process_wan21_latent_in/out`
- `apple_silicon_nodes/bridge.py` — application du de-whitening dans `_unpack_krea2_latents`
- `apple_silicon_nodes/sampler/core.py` — whitening du latent source pour Identity Edit
- `apple_silicon_nodes/native/krea2/model.py` — port de l'enhancer Krea2T
- `apple_silicon_nodes/sampler/scheduling.py` — fix du schedule sigma Krea2
- `apple_silicon_nodes/native/krea2/config.py`, `native/__init__.py`, `native/krea2/__init__.py` — nettoyage code mort

### Fichiers générés
- `graphify-out/` — graphe reconstruit (1109 nœuds, 1917 arêtes, 60 communautés)

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
