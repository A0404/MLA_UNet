import importlib
import os
import numpy as np
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage import distance_transform_edt
from unet_model import data_augmentation

# --------------------------------------------------
#  1. Fonction Center Crop Image
# --------------------------------------------------
def center_crop_img(feature_map, target_tensor_shape):
    """ Perform edge clipping in a centered manner."""
    h, w = feature_map.shape
    th, tw = target_tensor_shape

    delta_h = h - th
    delta_w = w - tw

    top = delta_h // 2
    left = delta_w // 2

    return feature_map[top:top+th, left:left+tw]


# ========== WEIGHTED CROSS ENTROPY LOSS ======================================
def unet_weight_map(mask, w0=10, sigma=5):
    """
    mask: numpy array (H, W), values {0,1}
    """
    labels = mask.astype(np.int32)

    # --- Class frequency ---
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts / counts.sum()

    # --- Class weights (UNet-style imbalance correction) ---
    # Option A (simple, robust)
    alpha = 1.0 / freq

    # Option B (more digitally stable, often preferable)
    # alpha = np.median(freq) / freq

    # --- Mapping class → weight ---
    alpha_map = dict(zip(classes, alpha))

    # --- Weight map ---
    w_c = np.zeros_like(labels, dtype=np.float32)
    for c, w in alpha_map.items():
        w_c[labels == c] = w

    # --- 2. Distance-based weights to separate touching objects ---
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

    # --- 3. Total Weight ---
    weight = w_c + w0 * np.exp(-((d1 + d2)**2) / (2 * sigma**2))
    weight = weight.astype(np.float32)

    """
    print("min(w_c) =", np.min(w_c), "; max(w_c) =", np.max(w_c), "; mean(w_c) =", np.mean(w_c))
    print("min(w0*exp(.)) =", np.min(w0 * np.exp(-((d1 + d2)**2) / (2 * sigma**2))), "; max(w0*exp(.)) =", np.max(w0 * np.exp(-((d1 + d2)**2) / (2 * sigma**2))), "; mean(w0*exp(.)) =", np.mean(w0 * np.exp(-((d1 + d2)**2) / (2 * sigma**2))))
    print("min(weight) =", np.min(weight), "; max(weight) =", np.max(weight), "; mean(weight) =", np.mean(weight),"\n") """

    return weight


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

        # Convert to numpy arrays
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        # Data augmentations
        if self.train:
            importlib.reload(data_augmentation)
            image, mask = data_augmentation.elastic_deformation_3x3(image, mask)
            image, mask = data_augmentation.random_rotate_shift(image, mask)
            image = data_augmentation.intensity_variation(image)

        # Resize mask to target size for UNet
        mask = center_crop_img(mask, (388, 388))

        # Transformations to tensors
        image = self.to_tensor(image / 255.0).float()

        # --- Prepare mask for weight map generation ---
        mask_np_for_wmap = mask.copy()  # copy original mask as numpy array
        mask_np_for_wmap = mask_np_for_wmap.astype(np.int32)  # convert to int32
        # Keep 0/255 values for distance-based weight calculation
        weight_map = unet_weight_map(mask_np_for_wmap)  # numpy float32
        weight_map = torch.from_numpy(weight_map).float()  # convert to tensor float

        # --- Binarize mask for the network ---
        mask = (mask == 255).astype(np.int32)  # binarize
        mask = torch.from_numpy(mask).long()  # convert to tensor long

        return (image, mask, weight_map)
    

# ========== DATASET CREATION =====================================
def dataset_ds(root_dir_1, root_dir_2, ratios=(0.7, 0.15, 0.15), seed=42):
    if root_dir_1 != root_dir_2:
        # Combine datasets from two directories
        all_pngs_1 = sorted(glob(os.path.join(root_dir_1, "*.png")))
        img2mask_1 = [(p, p.replace(".png", "_combined_mask.png")) 
                for p in all_pngs_1 if not p.endswith("_combined_mask.png")]
        
        all_pngs_2 = sorted(glob(os.path.join(root_dir_2, "*.png")))
        img2mask_2 = [(p, p.replace(".png", "_combined_mask.png")) 
                for p in all_pngs_2 if not p.endswith("_combined_mask.png")]
        
        # 2. Shuffle (NECESSARY)
        random.seed(seed)
        random.shuffle(img2mask_1)
        random.shuffle(img2mask_2)

        n_1 = len(img2mask_1)
        n_2 = len(img2mask_2)
        n_train = int(ratios[0] * n_1)
        print(f"Dataset split: {n_train} train | {n_1 - n_train} val | {n_2} test")

        return {"train":img2mask_1[:n_train], "val":img2mask_1[n_train:], "test":  img2mask_2,
            "seed": seed, "ratio": ratios}

    else:
        all_pngs = sorted(glob(os.path.join(root_dir_1, "*.png")))
        img2mask = [(p, p.replace(".png", "_combined_mask.png")) 
                    for p in all_pngs if not p.endswith("_combined_mask.png")]
                    
        # 2. Shuffle (NECESSARY)
        random.seed(seed)
        random.shuffle(img2mask)

        n = len(img2mask)
        n_train = int(ratios[0] * n)
        n_val   = int(ratios[1] * n)
        print(f"Dataset split: {n_train} train | {n_val} val | {n - n_train - n_val} test")

        return {"train":img2mask[:n_train], "val":img2mask[n_train:n_train+n_val], "test":  img2mask[n_train+n_val:],
            "seed": seed, "ratio": ratios}