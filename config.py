"""
Configuration paths for the MLA project.
Paths are computed relative to the project root.
"""
import os

# Project root (parent of this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Subfolders
DATASET_NORMALIZER_DIR = os.path.join(PROJECT_ROOT, "dataset_normalizer")
UNET_MODEL_DIR         = os.path.join(PROJECT_ROOT, "unet_model")
NOTEBOOKS_DIR          = os.path.join(PROJECT_ROOT, "notebooks")
SAVED_MODELS_DIR       = os.path.join(PROJECT_ROOT, "saved_models")

# Model save
UNet_BUSI_benign       = "unet_BUSI_benign.pth"
UNet_BUSI_malignant    = "unet_BUSI_malignant.pth"
UNet_ISBI              = "unet_isbi.pth"

# Flags pour tests
USE_DATA_AUG = False         # activer / désactiver DA
USE_IGNORE_INDEX = False     # activer / désactiver ignore label
USE_LOSS_POND = False        # activer / désactiver pondération label

DEBUG_VIS = True             # sauvegarde d’images
