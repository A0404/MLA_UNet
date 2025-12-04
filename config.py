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

# Model save paths (will be created if missing)
UNet_BUSI_benign       = os.path.join(UNET_MODEL_DIR, "unet_BUSI_benign.pth")
UNet_BUSI_malignant    = os.path.join(UNET_MODEL_DIR, "unet_BUSI_malignant.pth")
UNet_ISBI              = os.path.join(UNET_MODEL_DIR, "unet_isbi.pth")
