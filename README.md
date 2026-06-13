# UNet — Segmentation Cellulaire et Membranaire

Une implémentation PyTorch de l'architecture **U-Net** pour la segmentation d'images biomédicales, appliquée à trois benchmarks de microscopie : segmentation de membranes en microscopie électronique (ISBI), suivi cellulaire en contraste de phase (PhC) et suivi cellulaire DIC.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Structure du projet](#structure-du-projet)
- [Datasets](#datasets)
- [Installation](#installation)
- [Prétraitement des données](#prétraitement-des-données)
- [Entraînement](#entraînement)
- [Évaluation](#évaluation)
- [Architecture](#architecture)
- [Métriques](#métriques)
- [Configuration](#configuration)

---

## Vue d'ensemble

Ce projet réimplémente le pipeline de segmentation U-Net from scratch, avec un focus sur :

- **Architecture fidèle** : encodeur–décodeur avec skip connections, convolutions sans padding et center-crop, exactement comme décrit dans l'article original.
- **Support multi-datasets** : un seul pipeline gère les datasets BUSI, ISBI, PhC et DIC avec différentes structures d'entrée.
- **Loss pondérée** : weight maps de style UNet combinant correction du déséquilibre de classes et pondération basée sur la distance aux frontières.
- **Prétraitement robuste** : combinaison automatique des masques, normalisation du contraste (CLAHE + étirement par percentiles), et renommage cohérent des fichiers.
- **Augmentation au test** : moyenne sur 7 rotations pour la segmentation EM.

---

## Structure du projet

```
.
├── notebook.ipynb                  # Pipeline principal : entraînement, évaluation, analyse
├── organized_folder.ipynb          # Notebook de prétraitement des datasets
│
├── unet_model/
│   ├── model.py                    # Architecture U-Net (Encoder, Decoder, DoubleConv)
│   ├── dataset.py                  # Dataset, weight map, split train/val/test
│   ├── train.py                    # Boucle d'entraînement, early stopping, checkpointing
│   ├── test.py                     # Fonctions d'évaluation (EM + Cell Tracking)
│   ├── metrics.py                  # Pixel error, Rand error, Warping error, Dice, IoU
│   └── data_augmentation.py        # Déformation élastique, rotation/translation, variation d'intensité
│
└── dataset_normalizer/
    ├── name_and_folder.py          # Renommage et copie des fichiers avec nommage cohérent
    ├── resize_and_combine.py       # Redimensionnement et combinaison des masques multiples
    ├── contrast_normalizer.py      # Pipeline de normalisation du contraste (CLAHE)
    └── image_analysis.py           # Statistiques d'image et comparaison de méthodes de contraste
```

---

## Datasets

Le pipeline supporte les datasets suivants nativement :

| Dataset | Tâche | Stratégie de split |
|---------|-------|-------------------|
| **BUSI Bénin** | Segmentation échographie mammaire | 70/15/15 |
| **BUSI Malin** | Segmentation échographie mammaire | 70/15/15 |
| **ISBI** | Segmentation de membranes EM | 70/15/15 |
| **PhC** (GT + ST) | Suivi cellulaire contraste de phase | Entraînement sur ST, test sur GT |
| **DIC** (GT + ST) | Suivi cellulaire DIC | Entraînement sur ST, test sur GT |

Les datasets bruts doivent être placés dans les répertoires définis dans `config.py`. Formats d'entrée supportés : `.png`, `.jpg`, `.jpeg`, `.tif`.

---

## Installation

```bash
pip install torch torchvision Pillow opencv-python numpy scikit-image scipy scikit-learn
```

Python 3.8+ est recommandé. Un GPU compatible CUDA est fortement conseillé pour l'entraînement.

---

## Prétraitement des données

Le pipeline de prétraitement se déroule en trois étapes séquentielles via `organized_folder.ipynb` :

### 1. Renommage des fichiers (`name_and_folder.py`)

Normalise les noms de fichiers depuis les formats bruts hétérogènes vers un schéma cohérent :

```
img_000.png
img_000_mask_1.png
img_000_mask_2.png   ← plusieurs masques par image supportés
img_001.png
...
```

Deux structures d'entrée sont supportées :
- **Dossier unique** : images et masques coexistent, les masques sont identifiés par `_mask` dans le nom de fichier.
- **Deux dossiers** : répertoires séparés pour les images et les vérités terrain (ex : ISBI, PhC, DIC).

### 2. Redimensionnement et combinaison des masques (`resize_and_combine.py`)

- Redimensionne toutes les images et masques à **572×572** (taille d'entrée UNet).
- Combine les masques multiples par image en les sommant puis en binarisant (seuil 0.5).
- Génère un fichier `_combined_mask.png` par image.

### 3. Normalisation du contraste (`contrast_normalizer.py`) *(optionnel)*

Applique un pipeline d'amélioration en trois étapes aux images :

1. **Normalisation par la moyenne** — recentre l'histogramme sur mean = 0.5.
2. **Étirement du contraste par percentiles** — mappe l'intervalle [2e, 98e percentile] vers [0, 1].
3. **CLAHE** — égalisation adaptative locale de l'histogramme (clipLimit=2.0, tuile 8×8).

Les masques sont copiés sans modification.

Pour activer la normalisation, mettre `normalize = True` dans `organized_folder.ipynb`.

---

## Entraînement

L'entraînement se lance depuis `notebook.ipynb`, section **III.1**, ou directement via `train.py` :

```python
from unet_model.train import train_model

model, train_losses, val_losses, recall_liste, val_recall = train_model(
    model_dir    = "saved_models/",
    model_save   = "UNet_ISBI.pth",
    train_ds     = train_ds,
    val_ds       = val_ds,
    device       = device,
    resume       = False,
    hyperparams  = (1000, 3e-4, 0.99, 0.05, 1)
    # (num_epochs, lr, momentum, dropout_rate, batch_size)
)
```

### Fonctionnalités clés de l'entraînement

**Fonction de loss** — Cross-Entropie pondérée combinant :
- Correction du déséquilibre de classes (poids inversement proportionnels à la fréquence).
- Pondération basée sur la distance aux frontières : les pixels proches de cellules jointives reçoivent un poids de loss plus élevé (weight map UNet, `w0=10`, `sigma=5`).

**Optimiseur** — SGD avec momentum (défaut : lr=3e-4, momentum=0.99).

**Early stopping** (`EarlyStoppingFrontier`) :
- Surveille `val_loss / recall_frontier` comme critère d'optimisation.
- 30 époques de warmup avant qu'un arrêt soit possible.
- Ignore les améliorations si la fraction de frontière prédite tombe sous 8% (détection de modèle dégénéré).
- Patience : 20 époques.

**Checkpointing** — Le modèle, l'état de l'optimiseur et l'historique complet de l'entraînement sont sauvegardés après chaque époque dans un fichier `.pth`. L'entraînement peut être repris avec `resume=True`.

**Images de debug** — Si `DEBUG_VIS=True` dans `config.py`, des aperçus des prédictions sont sauvegardés toutes les 10 époques dans `debug_images/`.

### Augmentation des données

Appliquée à la volée pendant l'entraînement :

| Transformation | Paramètres |
|----------------|------------|
| Déformation élastique (grille 3×3) | sigma=10, p=0.7 |
| Rotation aléatoire + translation | ±15°, ±3px, p=0.8 |
| Gain + biais d'intensité | gain ∈ [0.9, 1.1], biais ∈ [–0.05, 0.05] |

Les masques utilisent une interpolation au plus proche voisin lors de toutes les transformations spatiales pour préserver les labels binaires.

---

## Évaluation

### Segmentation EM — ISBI (`test_em_unet`)

Conçue pour les benchmarks de segmentation de membranes. Utilise une **augmentation au test sur 7 rotations** : le modèle prédit sur les rotations [0°, 45°, 90°, 135°, 180°, 225°, 270°] et moyenne les cartes de probabilité après contre-rotation.

Le seuil optimal est sélectionné par balayage sur [0.20, 0.29] en minimisant le Rand Error.

```python
mean_warp, mean_rand, mean_pix, mean_prec, mean_rec = test.test_em_unet(
    save_path, test_ds, device
)
```

**Métriques de sortie :** Warping Error, Rand Error, Pixel Error, Précision, Rappel.

### Suivi cellulaire — PhC / DIC (`test_cell_tracking_unet`)

Utilise un **post-traitement par watershed** sur la transformée de distance de la carte de probabilité de premier plan pour séparer les cellules jointives.

```python
mean_iou, mean_dice, mean_prec, mean_rec = test.test_cell_tracking_unet(
    save_path, test_ds, device, threshold=0.25
)
```

**Métriques de sortie :** IoU, Dice, Précision, Rappel.

---

## Architecture

Le modèle est un U-Net standard avec la progression de canaux suivante :

```
Entrée (1×572×572)
    └─ Encodeur
          enc1 : 1   → 64    (→ 568×568)
          enc2 : 64  → 128   (→ 280×280)
          enc3 : 128 → 256   (→ 136×136)
          enc4 : 256 → 512   (→ 64×64)
          Dropout2d (p=0.05)
          goulot : 512 → 1024 (→ 28×28)
    └─ Décodeur (avec skip connections par center-crop)
          up1 + dec1 : 1024 → 512
          up2 + dec2 : 512  → 256
          up3 + dec3 : 256  → 128
          up4 + dec4 : 128  → 64
    └─ Conv finale 1×1 : 64 → num_classes (défaut 2)

Sortie (2×388×388)
```

Toutes les convolutions utilisent un noyau 3×3 sans padding. Les skip connections utilisent le center-crop pour aligner les dimensions spatiales. Les poids sont initialisés avec l'**initialisation He (Kaiming)**.

---

## Métriques

| Métrique | Description | Tâche |
|----------|-------------|-------|
| **Pixel Error** | Fraction de pixels mal classifiés | EM |
| **Rand Error** | 1 − Adjusted Rand Index sur la segmentation induite | EM |
| **Warping Error** | Distance moyenne de chaque pixel de frontière GT au pixel prédit le plus proche (normalisée) | EM |
| **IoU** | Intersection sur Union | Suivi cellulaire |
| **Dice** | 2·\|P∩G\| / (\|P\|+\|G\|) | Suivi cellulaire |
| **Précision / Rappel** | Précision et rappel pixel-level standards | Les deux |

---

## Configuration

Tous les chemins et les flags d'entraînement sont centralisés dans `config.py` (à adapter à votre configuration locale). Variables clés :

```python
# Répertoires des datasets bruts
RAW_BUSI_BENIGN_DATASET_DIR    = "..."
RAW_ISBI_IMG_DATASET_DIR       = "..."
RAW_ISBI_GT_DATASET_DIR        = "..."
# ... (PhC, DIC de la même façon)

# Répertoires de sortie des datasets normalisés
BUSI_BENIGN_DATASET_DIR        = "..."
ISBI_DATASET_DIR               = "..."
# ...

# Chemins de sauvegarde des modèles
SAVED_MODELS_DIR               = "saved_models/"
UNet_ISBI                      = "UNet_ISBI.pth"
# ...

# Flag de debug
DEBUG_VIS = False   # Mettre à True pour sauvegarder les images de prédiction toutes les 10 époques
```
