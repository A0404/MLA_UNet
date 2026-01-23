# MLA_UNet - Segmentation de Membranes Cellulaires

Implémentation PyTorch d'un réseau U-Net pour la segmentation de membranes cellulaires en microscopie électronique, optimisé pour le challenge ISBI 2012.


##  Aperçu

Ce projet implémente l'architecture U-Net, un réseau de neurones convolutif spécialement conçu pour la segmentation d'images biomédicales. L'objectif principal est de segmenter avec précision les membranes cellulaires dans des images de microscopie électronique.

### Caractéristiques principales

- **Architecture U-Net classique** avec encoder-decoder
- **Augmentation de données** inspirée du papier original U-Net
- **Fonction de perte pondérée** pour gérer le déséquilibre de classes
- **Early stopping intelligent** basé sur la qualité des frontières
- **Système de checkpointing** pour reprendre l'entraînement
- **Métriques d'évaluation complètes** (Dice, IoU, Rand Error, Warping Error)

##  Fonctionnalités

### Augmentation de données

- **Déformations élastiques** : Grille 3×3 avec interpolation bicubique (σ=10, p=0.7)
- **Rotations aléatoires** : Angles jusqu'à ±15° (p=0.8)
- **Décalages aléatoires** : Translations jusqu'à ±3 pixels
- **Variations d'intensité** : Gain [0.9, 1.1] et biais [-0.05, 0.05]

### Fonction de perte

- **Cross-Entropy pondérée** avec deux composantes :
  - Pondération par fréquence de classe (déséquilibre)
  - Pondération par distance (séparation des objets voisins)
- Formule : `w(x) = w_c(x) + w_0 * exp(-((d1 + d2)²) / (2σ²))`

### Early Stopping

Stratégie intelligente basée sur :
- **Warmup** : 30 époques minimum avant arrêt possible
- **Recall frontal** : Suivi de la détection des membranes
- **Fraction de frontière** : Vérification de cohérence structurelle (min. 8%)
- **Patience** : 20 époques sans amélioration

## Architecture

```
                Input (572x572x1)
                       |
        ┌──────────────┴──────────────┐
        │         ENCODER              │
        │  ┌─────────────────────┐    │
        │  │  64 channels        │────┤ skip1
        │  └─────────────────────┘    │
        │           ↓ pool             │
        │  ┌─────────────────────┐    │
        │  │  128 channels       │────┤ skip2
        │  └─────────────────────┘    │
        │           ↓ pool             │
        │  ┌─────────────────────┐    │
        │  │  256 channels       │────┤ skip3
        │  └─────────────────────┘    │
        │           ↓ pool             │
        │  ┌─────────────────────┐    │
        │  │  512 channels       │────┤ skip4
        │  └─────────────────────┘    │
        │           ↓ pool             │
        │  ┌─────────────────────┐    │
        │  │  1024 channels      │    │
        │  │  (Bottleneck)       │    │
        │  └─────────────────────┘    │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴──────────────┐
        │         DECODER              │
        │  ┌─────────────────────┐    │
        │  │  512 channels       │←───┤ skip4 (crop + concat)
        │  └─────────────────────┘    │
        │           ↑ upsample         │
        │  ┌─────────────────────┐    │
        │  │  256 channels       │←───┤ skip3
        │  └─────────────────────┘    │
        │           ↑ upsample         │
        │  ┌─────────────────────┐    │
        │  │  128 channels       │←───┤ skip2
        │  └─────────────────────┘    │
        │           ↑ upsample         │
        │  ┌─────────────────────┐    │
        │  │  64 channels        │←───┤ skip1
        │  └─────────────────────┘    │
        └──────────────┬───────────────┘
                       │
              Output (388x388x2)
```

### Détails de l'architecture

- **Convolutions** : Kernel 3×3, sans padding (valid convolution)
- **Activation** : ReLU
- **Pooling** : MaxPool 2×2
- **Upsampling** : ConvTranspose2d stride 2
- **Initialisation** : He initialization pour toutes les couches
- **Dropout** : 5% après le dernier pooling (optionnel)
- **Skip connections** : Center crop avant concaténation

##  Installation

### Prérequis

- Python 3.8+
- CUDA compatible GPU

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/A0404/MLA_UNet.git
cd MLA_UNet

# Installer les dépendances
pip install torch torchvision
pip install numpy pillow scipy scikit-image scikit-learn opencv-python matplotlib
```

### Liste complète des dépendances

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pillow>=8.0.0
scipy>=1.5.0
scikit-image>=0.18.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
matplotlib>=3.3.0
```

## Structure du projet

```
MLA_UNet/
│
├── model.py                 # Architecture U-Net
├── dataset.py               # Dataset et preprocessing
├── data_augmentation.py     # Augmentation de données
├── train.py                 # Script d'entraînement
├── test.py                  # Script d'évaluation
├── metrics.py               # Métriques d'évaluation
├── notebook__1_.ipynb       # Notebook de démonstration
│
├── config.py                # Configuration (à créer)
└── README.md                # Ce fichier
```

##  Utilisation

### 1. Préparation des données

Organisez vos données comme suit :

```
data/
├── train/
│   ├── image_001.png
│   ├── image_001_combined_mask.png
│   ├── image_002.png
│   ├── image_002_combined_mask.png
│   └── ...
└── test/
    ├── image_test_001.png
    ├── image_test_001_combined_mask.png
    └── ...
```

**Note** : Les masques doivent avoir le suffixe `_combined_mask.png` et contenir :
- `0` pour les membranes
- `255` pour les cellules

### 2. Configuration

Créez un fichier `config.py` :

```python
# Chemins de données
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Hyperparamètres
NUM_EPOCHS = 1000
LEARNING_RATE = 3e-4
MOMENTUM = 0.99
DROPOUT_RATE = 0.05
BATCH_SIZE = 1

# Options
DEBUG_VIS = True  # Sauvegarder des images de débogage
MODEL_DIR = "models"
MODEL_NAME = "unet_checkpoint.pth"
```

### 3. Entraînement

```python
from dataset import dataset_ds, SegmentationDataset
from train import train_model
import torch

# Créer les datasets
data_dict = dataset_ds(
    root_dir_1=TRAIN_DIR,
    root_dir_2=TEST_DIR,
    ratios=(0.7, 0.15, 0.15),
    seed=42
)

train_ds = SegmentationDataset(dict(data_dict["train"]), train=True)
val_ds = SegmentationDataset(dict(data_dict["val"]), train=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamètres
hyperparams = (NUM_EPOCHS, LEARNING_RATE, MOMENTUM, DROPOUT_RATE, BATCH_SIZE)

# Entraînement
model, train_losses, val_losses, recall_list, val_recall = train_model(
    model_dir=MODEL_DIR,
    model_save=MODEL_NAME,
    train_ds=train_ds,
    val_ds=val_ds,
    device=device,
    resume=False,  # Mettre True pour reprendre l'entraînement
    hyperparams=hyperparams
)
```

### 4. Reprise de l'entraînement

```python
# Pour reprendre l'entraînement depuis un checkpoint
model, train_losses, val_losses, recall_list, val_recall = train_model(
    model_dir=MODEL_DIR,
    model_save=MODEL_NAME,
    train_ds=train_ds,
    val_ds=val_ds,
    device=device,
    resume=True,  # Active la reprise
    hyperparams=hyperparams  # Sera écrasé par les hyperparamètres du checkpoint
)
```

### 5. Évaluation

```python
from test import evaluate_model

# Charger le modèle entraîné
checkpoint = torch.load("models/unet_checkpoint.pth")
model = UNet(dropout_rate=0.05)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)

# Créer le dataset de test
test_ds = SegmentationDataset(dict(data_dict["test"]), train=False)

# Évaluer
results = evaluate_model(model, test_ds, device)

print(f"Dice Score: {results['dice']:.4f}")
print(f"IoU Score: {results['iou']:.4f}")
print(f"Pixel Error: {results['pixel_error']:.4f}")
print(f"Rand Error: {results['rand_error']:.4f}")
print(f"Warping Error: {results['warping_error']:.4f}")
```

## 🔧 Détails techniques

### Preprocessing

1. **Normalisation** : Images divisées par 255 → [0, 1]
2. **Center crop** : Masques recadrés à 388×388 (sortie du réseau)
3. **Binarisation** : Masques convertis en {0, 1}

### Fonction de perte pondérée

La fonction de perte combine deux stratégies :

1. **Pondération par classe** : `w_c(x) = 1 / freq_class`
   - Compense le déséquilibre entre membranes et cellules

2. **Pondération par distance** : `w_0 * exp(-((d1 + d2)²) / (2σ²))`
   - `d1`, `d2` : distances aux 2 objets les plus proches
   - `w_0 = 10` : importance de la séparation
   - `σ = 5` : largeur de la zone de séparation

### Optimisation

- **Optimizer** : SGD avec momentum
- **Learning rate** : 3×10⁻⁴ (par défaut)
- **Momentum** : 0.99
- **Batch size** : 1 (recommandé pour les petits datasets)

### Checkpointing

Chaque checkpoint contient :
- État du modèle (`model_state`)
- État de l'optimiseur (`optimizer_state`)
- Hyperparamètres utilisés
- Historique des pertes (train/val)
- Historique des métriques
- État du early stopping

##  Métriques d'évaluation

### Métriques implémentées

1. **Dice Coefficient** : Mesure de similarité (0-1, plus haut = mieux)
   ```
   Dice = 2|X ∩ Y| / (|X| + |Y|)
   ```

2. **IoU (Intersection over Union)** : Ratio d'intersection (0-1)
   ```
   IoU = |X ∩ Y| / |X ∪ Y|
   ```

3. **Pixel Error** : Erreur de classification pixel par pixel
   ```
   PE = 1 - accuracy
   ```

4. **Rand Error** : Erreur de segmentation basée sur Rand Index
   - Compare les régions connexes
   - Invariant aux étiquettes

5. **Warping Error** : Distance moyenne des frontières
   - Mesure la précision du contour
   - Normalisé par la diagonale de l'image

### Métriques d'entraînement

- **Recall frontal** : Taux de détection des membranes
- **Frontier fraction** : Proportion de pixels membrane prédits
- **Loss/Recall ratio** : Indicateur de qualité global

## Résultats

### Configuration optimale

Les meilleurs résultats sont généralement obtenus avec :

- **Learning rate** : 3×10⁻⁴
- **Dropout** : 5%
- **Batch size** : 1
- **Augmentation** : Toutes activées
- **Epochs** : ~300-500 (avec early stopping)

### Performance attendue

Sur le dataset ISBI 2012 :
- **Dice** : > 0.85
- **IoU** : > 0.75
- **Warping Error** : < 0.001 (normalisé)

*Note : Les résultats dépendent fortement de la qualité des annotations et de la taille du dataset.*

## Bonnes pratiques

### Pour l'entraînement

1. **Commencer simple** : Entraîner sans dropout ni augmentation
2. **Ajouter progressivement** : Tester l'impact de chaque technique
3. **Monitorer** : Surveiller les métriques ET les images de débogage
4. **Patience** : Laisser le warmup complet (30 epochs minimum)

### Pour le debugging

1. Activer `DEBUG_VIS = True` dans config
2. Vérifier les images générées dans `debug_images/`
3. S'assurer que les prédictions convergent vers des membranes cohérentes
4. Surveiller la fraction de frontière (doit rester > 8%)

### Problèmes courants

| Problème | Solution |
|----------|----------|
| Modèle prédit tout à 0 ou 1 | Vérifier les poids de la loss, réduire le learning rate |
| Frontières trop épaisses | Augmenter `w_0` dans la weight map |
| Underfitting | Augmenter le nombre d'epochs, réduire le dropout |
| Overfitting | Augmenter l'augmentation de données, ajouter du dropout |


##  Références

- **U-Net Paper** : [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **ISBI Challenge** : [Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)
- **PyTorch Documentation** : [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)

##  Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

