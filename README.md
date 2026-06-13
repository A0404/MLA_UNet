# MLA_UNet — Segmentation de Membranes Cellulaires

Une implémentation complète du réseau **U-Net** pour la segmentation sémantique d'images biomédicales (membranes cellulaires), incluant le prétraitement des données, l'entraînement, l'évaluation et l'inférence.

---

## Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Structure du dépôt](#structure-du-dépôt)
- [Datasets supportés](#datasets-supportés)
- [Installation](#installation)
- [Configuration](#configuration)
- [Pipeline de préparation des données](#pipeline-de-préparation-des-données)
- [Entraînement](#entraînement)
- [Évaluation](#évaluation)
- [Modèle pré-entraîné (MONAI)](#modèle-pré-entraîné-monai)
- [Architecture U-Net](#architecture-u-net)
- [Métriques](#métriques)

---

## Aperçu du projet

Ce projet implémente le U-Net original (Ronneberger et al., 2015) pour segmenter des structures biologiques (membranes, cellules) dans des images microscopiques. Il prend en charge plusieurs datasets publics et intègre :

- Un pipeline de normalisation et de préparation des données
- De l'augmentation de données (déformation élastique, rotation, variation d'intensité)
- Une fonction de perte pondérée (Weighted Cross-Entropy) inspirée du papier original
- Un early stopping basé sur la qualité des contours détectés
- Un support multi-datasets (BUSI, ISBI, PhC, DIC)

---

## Structure du dépôt

```
MLA_UNet/
│
├── bdd/
│   ├── non_normalized/          # Données brutes (à fournir par l'utilisateur)
│   │   ├── BUSI/benign/
│   │   ├── BUSI/malignant/
│   │   ├── ISBI/{Img,GT}/
│   │   ├── PhC/{Img,GT,ST}/
│   │   └── DIC/{Img,GT,ST}/
│   └── normalized/              # Données prétraitées (générées automatiquement)
│
├── dataset_normalizer/          # Pipeline de préparation des données
│   ├── contrast_normalizer.py   # Normalisation du contraste (CLAHE, percentile)
│   ├── image_analysis.py        # Analyse et comparaison de méthodes de contraste
│   ├── name_and_folder.py       # Renommage et organisation des fichiers
│   └── resize_and_combine.py    # Redimensionnement et fusion des masques
│
├── notebooks/
│   ├── notebook.ipynb           # Notebook principal d'expérimentation
│   └── organized_folder.ipynb   # Notebook de préparation des datasets
│
├── pretrained_model/
│   ├── model_pretrained.py      # Entraînement avec MONAI (baseline)
│   └── code_de_test_model.py    # Test rapide d'un modèle MONAI
│
├── unet_model/                  # Implémentation principale
│   ├── model.py                 # Architecture U-Net (Encoder, Decoder, DoubleConv)
│   ├── dataset.py               # Dataset PyTorch + weight map UNet
│   ├── data_augmentation.py     # Augmentations (élastique, rotation, intensité)
│   ├── metrics.py               # Métriques d'évaluation
│   ├── train.py                 # Boucle d'entraînement complète
│   └── test.py                  # Évaluation sur données de test
│
├── saved_models/                # Checkpoints sauvegardés (.pth)
├── config.py                    # Centralisation des chemins et hyperparamètres
├── requirements.txt             # Dépendances Python
└── README.md
```

---

## Datasets supportés

| Dataset | Type | Description |
|---|---|---|
| **BUSI Benign** | Échographie | Tumeurs mammaires bénignes |
| **BUSI Malignant** | Échographie | Tumeurs mammaires malignes |
| **ISBI** | Microscopie EM | Segmentation de membranes (challenge ISBI 2012) |
| **PhC-GT / PhC-ST** | Contraste de phase | Suivi de cellules (Cell Tracking Challenge) |
| **DIC-GT / DIC-ST** | DIC | Suivi de cellules (Cell Tracking Challenge) |

Les données brutes doivent être placées dans `bdd/non_normalized/` selon la structure décrite ci-dessus.

---

## Installation

**Prérequis :** Python 3.8+, pip

```bash
# Cloner le dépôt
git clone https://github.com/amine-laroussi/MLA_UNet.git
cd MLA_UNet

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances principales :**

```
torch
torchvision
Pillow
opencv-python
numpy
scikit-image
scipy
scikit-learn
```

Pour le module `pretrained_model`, MONAI est également requis :

```bash
pip install monai
```

---

## Configuration

Tous les chemins et hyperparamètres sont centralisés dans `config.py`. Les chemins sont calculés automatiquement à partir de la racine du projet.

```python
# config.py — variables principales

# Répertoires des données brutes
RAW_BUSI_BENIGN_DATASET_DIR    = "bdd/non_normalized/BUSI/benign"
RAW_ISBI_IMG_DATASET_DIR       = "bdd/non_normalized/ISBI/Img"
# ...

# Répertoires des données normalisées
BUSI_BENIGN_DATASET_DIR        = "bdd/normalized/BUSI_benign"
ISBI_DATASET_DIR               = "bdd/normalized/ISBI"
# ...

# Noms des modèles sauvegardés
UNet_BUSI_benign    = "unet_BUSI_benign.pth"
UNet_ISBI           = "unet_isbi.pth"

# Visualisation debug (sauvegarde d'images toutes les 10 époques)
DEBUG_VIS = True
```

---

## Pipeline de préparation des données

La préparation se fait via le notebook `notebooks/organized_folder.ipynb` ou directement par les scripts du dossier `dataset_normalizer/`.

### Étapes du pipeline

**1. Renommage et organisation** (`name_and_folder.py`)

Copie et renomme les fichiers images/masques selon un format cohérent :
```
img_000.png, img_000_mask_1.png, img_001.png, img_001_mask_1.png, ...
```

**2. Redimensionnement et fusion des masques** (`resize_and_combine.py`)

- Redimensionne les images à **572×572** pixels
- Fusionne les masques multiples par image en un masque binaire unique (`_combined_mask.png`)

**3. Normalisation du contraste** (`contrast_normalizer.py`) *(optionnel)*

Applique successivement :
- Recentrage de la moyenne (`normalize_to_mean`, cible = 0.5)
- Étirement des percentiles 2–98 (`contrast_stretch`)
- CLAHE via OpenCV (`clipLimit=2.0`, `tileGridSize=8×8`)

### Utilisation

```python
# Dans organized_folder.ipynb
normalize = False   # Mettre True pour activer la normalisation du contraste

for i in range(len(output_dir)):
    name_and_folder.name_in_folder(input_dir[i], temp_dir)
    resize_and_combine.mask_combiner(temp_dir, output_dir[i])
    # Si normalize=True : contrast_normalizer.contrast(...)
```

---

## Entraînement

### Lancement

```python
from config import SAVED_MODELS_DIR, ISBI_DATASET_DIR
from unet_model.dataset import dataset_ds, SegmentationDataset
from unet_model.train import train_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Créer les splits train/val/test
splits = dataset_ds(ISBI_DATASET_DIR, ISBI_DATASET_DIR, ratios=(0.7, 0.15, 0.15))

train_ds = SegmentationDataset(dict(splits["train"]), train=True)
val_ds   = SegmentationDataset(dict(splits["val"]),   train=False)

# Entraîner
model, train_losses, val_losses, recalls, val_recalls = train_model(
    model_dir   = SAVED_MODELS_DIR,
    model_save  = "unet_isbi.pth",
    train_ds    = train_ds,
    val_ds      = val_ds,
    device      = device,
    resume      = False,
    hyperparams = (1000, 3e-4, 0.99, 0.05, 1)
    #              epochs, lr,  momentum, dropout, batch_size
)
```

### Fonctionnalités d'entraînement

- **Perte** : Weighted Cross-Entropy (pondération par pixel via `unet_weight_map`)
- **Optimiseur** : SGD avec momentum
- **Checkpointing** : Sauvegarde automatique à chaque époque dans `saved_models/`
- **Reprise** : `resume=True` pour continuer un entraînement interrompu
- **Early stopping** : Basé sur le ratio `val_loss / recall_frontier` avec warmup (30 époques par défaut)
- **Debug** : Si `DEBUG_VIS=True` dans `config.py`, des images de prédiction sont sauvegardées toutes les 10 époques dans `debug_images/`

---

## Évaluation

Deux fonctions de test selon le type de dataset :

### Segmentation de membranes (EM / ISBI)

```python
from unet_model.test import test_em_unet

test_em_unet(
    save_path         = "saved_models/unet_isbi.pth",
    test_ds           = test_ds,
    device            = device,
    thresholds        = np.linspace(0.2, 0.29, 10),
    num_samples_to_show = 2
)
```

Utilise 7 rotations (0°, 45°, 90°, 135°, 180°, 225°, 270°) et moyenne les probabilités pour améliorer la robustesse.

### Suivi cellulaire (PhC / DIC / BUSI)

```python
from unet_model.test import test_cell_tracking_unet

test_cell_tracking_unet(
    save_path         = "saved_models/unet_phc.pth",
    test_ds           = test_ds,
    device            = device,
    threshold         = 0.25,
    num_samples_to_show = 2
)
```

Applique un **watershed** sur la carte de distance pour séparer les cellules jointives.

---

## Modèle pré-entraîné (MONAI)

Le dossier `pretrained_model/` contient une approche alternative utilisant l'implémentation U-Net de **MONAI** :

```bash
# Entraînement rapide (baseline MONAI)
python pretrained_model/model_pretrained.py

# Test visuel d'un modèle sauvegardé
python pretrained_model/code_de_test_model.py
```

Configurer les chemins directement dans les fichiers avant exécution.

---

## Architecture U-Net

L'architecture suit fidèlement le papier original (Ronneberger et al., 2015) :

```
Entrée (1×572×572)
        │
  ┌─────▼─────┐
  │  Encoder  │  4 niveaux : DoubleConv + MaxPool2d
  │           │  Filtres : 64 → 128 → 256 → 512
  ├─────▼─────┤
  │ Bottleneck│  DoubleConv(512 → 1024) + Dropout2d
  ├─────▼─────┤
  │  Decoder  │  4 niveaux : ConvTranspose2d + center_crop + concat + DoubleConv
  │           │  Filtres : 1024 → 512 → 256 → 128 → 64
  └─────▼─────┘
  Conv1×1 → 2 classes (fond / membrane)
  Sortie (2×388×388)
```

Points clés :
- **Convolutions sans padding** (`padding=0`) — fidèle au papier original
- **Center crop** des skip connections pour aligner les dimensions
- **Initialisation He** (`kaiming_normal_`) sur toutes les couches convolutives
- **Dropout2d** avant le bottleneck (taux configurable, défaut 5%)
- **Sortie 388×388** (entrée 572×572 avec les 4 niveaux sans padding)

---

## Métriques

| Métrique | Fonction | Usage |
|---|---|---|
| **Pixel Error** | `pixel_error(pred, gt)` | Taux d'erreur pixel à pixel |
| **Rand Error** | `rand_error(pred, gt)` | Erreur sur la segmentation induite (ISBI) |
| **Warping Error** | `warping_error_normalized(pred, gt)` | Distance aux contours (ISBI) |
| **Dice** | `dice(pred, gt)` | Chevauchement binary mask |
| **IoU** | `iou_score(pred, gt)` | Intersection over Union |
| **Precision / Recall** | calculées dans `test.py` | Précision et rappel sur la classe foreground |

---

## Augmentation des données

Trois types d'augmentation sont appliqués durant l'entraînement (`data_augmentation.py`) :

- **Déformation élastique** (`elastic_deformation_3x3`) — grille 3×3 avec interpolation bicubique, inspirée du papier U-Net original (p=0.7)
- **Rotation + translation aléatoire** (`random_rotate_shift`) — ±15° de rotation, ±3px de décalage (p=0.8)
- **Variation d'intensité** (`intensity_variation`) — gain ∈ [0.9, 1.1], biais ∈ [-0.05, 0.05]

---

## Références

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI 2015.
- [ISBI 2012 Challenge — EM Segmentation](http://brainiac2.mit.edu/isbi_challenge/)
- [Cell Tracking Challenge](http://celltrackingchallenge.net/)
- [BUSI Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
