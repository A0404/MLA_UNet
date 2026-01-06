import os
import numpy as np
import cv2
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage import distance_transform_edt
from dataset_normalizer.data_augmentation import elastic_deformation_3x3, random_rotate_shift, intensity_variation
from config import USE_DATA_AUG, USE_IGNORE_INDEX, USE_LOSS_POND

# ========== WEIGHTED CROSS ENTROPY LOSS ======================================
def unet_weight_map(mask, w0=10, sigma=5):
    """
    mask: numpy array (H, W), values {0,1}
    """
    labels = mask.astype(np.int32)

    # --- 1. Poids de classe pour corriger le déséquilibre ---
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = {c: 1.0/count for c, count in zip(unique, counts)}
    w_c = np.zeros_like(labels, dtype=np.float32)
    for c in unique:
        w_c[labels == c] = class_weights[c]

    # --- 2. Composante "bord" pour séparer les membranes proches ---
    if labels.max() < 2:
        # Si pas de classes multiples, poids uniforme
        weight = np.ones_like(labels, dtype=np.float32)
        return weight

    distances = []
    for label_id in range(1, labels.max()+1):
        distances.append(distance_transform_edt(labels != label_id))
    
    distances = np.stack(distances)
    d1 = np.min(distances, axis=0)
    d2 = np.partition(distances, 1, axis=0)[1]

    # --- 3. Poids total ---
    weight = w_c + w0 * np.exp(-((d1 + d2)**2) / (2 * sigma**2))
    weight = weight.astype(np.float32)

    weight_map = np.clip(weight, 1e-2, 1000)    # clip extreme values
    weight /= weight.mean()

    return weight_map


# ========== DATASET DEFINITION ================================================
class SegmentationDataset(Dataset):
    def __init__(self, img2mask, train=True, use_ignore_index=False):
        self.img2mask = img2mask
        self.images = list(img2mask.keys())
        self.train = train
        self.to_tensor = transforms.ToTensor()
        self.use_ignore_index = use_ignore_index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieving paths
        img_path = self.images[idx]
        mask_path = self.img2mask[img_path]

        # Retrieving images and masks
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Convert to numpy arrays
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        # Data augmentations
        if self.train:
            image, mask = elastic_deformation_3x3(image, mask)
            image, mask = random_rotate_shift(image, mask)
            image = intensity_variation(image)

        # Resize mask to target size for UNet
        mask = cv2.resize(mask, (388, 388), interpolation=cv2.INTER_NEAREST)

        # Transformations to tensors and mask binarization
        image = self.to_tensor(image).float()

        # --- Prepare mask for weight map generation ---
        mask_np_for_wmap = mask.copy()  # copy original mask as numpy array
        mask_np_for_wmap = mask_np_for_wmap.astype(np.int32)  # convert to int32
        # Keep 0/255 values for distance-based weight calculation
        weight_map = unet_weight_map(mask_np_for_wmap)  # numpy float32
        weight_map = torch.from_numpy(weight_map).float()  # convert to tensor float

        # --- Binarize mask for the network ---
        mask = mask.astype(np.int32)
        if self.use_ignore_index:
            mask[mask == 128] = 128  # keep ignore index intact
        mask[mask != 128] = (mask[mask != 128] == 255).astype(np.int32)  # binarize
        mask = torch.from_numpy(mask).long()  # convert to tensor long

        return (image, mask, weight_map)
    

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
    train_ds = SegmentationDataset(train_files, train=USE_DATA_AUG, use_ignore_index=USE_IGNORE_INDEX)
    val_ds = SegmentationDataset(train_files, train=False, use_ignore_index=USE_IGNORE_INDEX)
    test_ds = SegmentationDataset(test_files, train=False, use_ignore_index=USE_IGNORE_INDEX)

    return (train_ds, val_ds, test_ds)