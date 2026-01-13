"""
Configuration paths for the MLA project.
Paths are computed relative to the project root.
"""
import os

# Project root (parent of this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Subfolders
DATASET_NORMALIZER_DIR      = os.path.join(PROJECT_ROOT, "dataset_normalizer")
UNET_MODEL_DIR              = os.path.join(PROJECT_ROOT, "unet_model")
NOTEBOOKS_DIR               = os.path.join(PROJECT_ROOT, "notebooks")
SAVED_MODELS_DIR            = os.path.join(PROJECT_ROOT, "saved_models")

# Raw dataset paths
RAW_BUSI_BENIGN_DATASET_DIR       = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "BUSI", "benign")
RAW_BUSI_MALIGNANT_DATASET_DIR    = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "BUSI", "malignant")
RAW_ISBI_IMG_DATASET_DIR          = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "ISBI", "Img")
RAW_ISBI_GT_DATASET_DIR           = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "ISBI", "GT")
RAW_PHC_IMG_DATASET_DIR           = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "PhC", "Img")
RAW_PHC_GT_DATASET_DIR            = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "PhC", "GT")
RAW_PHC_ST_DATASET_DIR            = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "PhC", "ST")
RAW_DIC_IMG_DATASET_DIR           = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "DIC", "Img")
RAW_DIC_GT_DATASET_DIR            = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "DIC", "GT")
RAW_DIC_ST_DATASET_DIR            = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "DIC", "ST")

# Normalized dataset paths
BUSI_BENIGN_DATASET_DIR    = os.path.join(PROJECT_ROOT, "bdd", "normalized", "BUSI_benign")
BUSI_MALIGNANT_DATASET_DIR = os.path.join(PROJECT_ROOT, "bdd", "normalized", "BUSI_malignant")
ISBI_DATASET_DIR           = os.path.join(PROJECT_ROOT, "bdd", "normalized", "ISBI")
PHC_GT_DATASET_DIR         = os.path.join(PROJECT_ROOT, "bdd", "normalized", "PhC_GT")
PHC_ST_DATASET_DIR         = os.path.join(PROJECT_ROOT, "bdd", "normalized", "PhC_ST")
DIC_GT_DATASET_DIR         = os.path.join(PROJECT_ROOT, "bdd", "normalized", "DIC_GT")
DIC_ST_DATASET_DIR         = os.path.join(PROJECT_ROOT, "bdd", "normalized", "DIC_ST")

# Model save
UNet_BUSI_benign       = "unet_BUSI_benign.pth"
UNet_BUSI_malignant    = "unet_BUSI_malignant.pth"
UNet_ISBI              = "unet_isbi.pth"
UNet_PhC               = "unet_phc.pth"
UNet_DIC               = "unet_dic.pth"

# Save debug visualizations
DEBUG_VIS = True            
