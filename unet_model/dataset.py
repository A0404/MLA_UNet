import os
import numpy as np
import cv2
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataset_normalizer.data_augmentation import elastic_deformation_3x3, random_rotate_shift, intensity_variation


# ========== IMAGE NORMALIZATION FUNCTION =====================================
def normalize_image(img_np):
    """
    ISBI-style normalization:
    1. percentile clipping
    2. centering (subtract mean)
    3. normalization (divide by std)
    """

    # 1) intensity clipping
    low = np.percentile(img_np, 0.5)
    high = np.percentile(img_np, 99.5)
    img_np = np.clip(img_np, low, high)

    # 2) centering
    img_np = img_np - img_np.mean()

    # 3) normalization
    img_np = img_np / (img_np.std() + 1e-8)

    return img_np


# ========== DATASET DEFINITION ================================================
class SegmentationDataset(Dataset):
    def __init__(self, img2mask, train=True):
        self.img2mask = img2mask
        self.images = list(img2mask.keys())
        self.train = train
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieving paths
        img_path = self.images[idx]
        mask_path = self.img2mask[img_path]

        # Retrieving images and masks
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        if self.train:
            image, mask = elastic_deformation_3x3(image, mask)
            image, mask = random_rotate_shift(image, mask)
            image = intensity_variation(image)

        # === Apply ISBI-style normalization ===
        image = normalize_image(image)

        # --- Resize mask to target size for UNet ---
        mask = cv2.resize(mask, (388, 388), interpolation=cv2.INTER_NEAREST)

        # Transformations to tensors and mask binarization
        image = self.to_tensor(image).float()
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).long().squeeze(0)  # Binarize and remove channel dim

        return (image, mask)
    

# ========== DATASET CREATION =====================================
def dataset_ds(root_dir, ratio=0.8, seed=42):
    # Combine image and mask paths
    all_pngs = sorted(glob(os.path.join(root_dir, "*.png")))
    img2mask = [(p, p.replace(".png", "_combined_mask.png")) 
            for p in all_pngs if not p.endswith("_combined_mask.png")]
    
    # 2. Shuffle (OBLIGATOIRE)
    random.seed(seed)
    random.shuffle(img2mask)

    n = int(ratio*len(img2mask))
    train_files = dict(img2mask[:n])
    test_files = dict(img2mask[n:])
    
    #Datasets
    train_ds = SegmentationDataset(train_files, train=True)
    val_ds = SegmentationDataset(train_files, train=False)
    test_ds = SegmentationDataset(test_files, train=False)
    
    return (train_ds, val_ds,test_ds)