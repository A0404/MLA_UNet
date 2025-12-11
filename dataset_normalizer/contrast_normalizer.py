import os
import cv2
from PIL import Image
import numpy as np
from glob import glob

# ------------------- Normalization to mean -------------------
def normalize_to_mean(img, target_mean=0.5):
    """
    Normalizes each image to center its mean on target_mean
    img: np.float32 in [0,1]
    """
    a = img.astype(np.float32)
    mean = a.mean()
    a = a - mean + target_mean
    a = np.clip(a, 0.0, 1.0)
    return a

# ------------------- Contrast enhancement -------------------
def contrast_stretch(img, low_perc=2, high_perc=98):
    """Stretch the contrast between low_perc and high_perc percentiles"""
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, [low_perc, high_perc])
    if hi - lo < 1e-6:
        return a
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

# ------------------- CLAHE -------------------
def clahe_cv2(img, clipLimit=2.0, tileGridSize=(8,8)):
    """CLAHE via OpenCV. Input img in float32 [0,1], output float32 [0,1]"""
    scaled = np.clip(img*255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    out = clahe.apply(scaled)
    return out.astype(np.float32)/255.0

# ------------------- Main pipeline -------------------
def contrast(input_dir, output_dir):
    """
    input_dir: dossier contenant les images originales (pas les masks combinés)
    output_dir: dossier où sauvegarder les images normalisées
    """
    os.makedirs(output_dir, exist_ok=True)

    # Récupérer tous les PNG
    all_pngs = sorted(glob(os.path.join(input_dir, "*.png")))
    img_paths = []
    mask_paths = []
    for p in all_pngs:
        if "_combined_mask.png" not in os.path.basename(p):
            img_paths.append(p)
        else:
            mask_paths.append(p)

    combined_count = 0
    for path in img_paths:
        # Lire avec PIL
        img = Image.open(path).convert('L')
        img = np.array(img).astype(np.float32)/255.0

        # Normalisation et contrast enhancement
        img = normalize_to_mean(img, target_mean=0.5)
        img = contrast_stretch(img, 2, 98)
        img = clahe_cv2(img, clipLimit=2.0, tileGridSize=(8,8))

        # Sauvegarde
        base_name = os.path.basename(path)
        out_path = os.path.join(output_dir, base_name)
        Image.fromarray((img*255).astype(np.uint8)).save(out_path)

        combined_count += 1

    for path in mask_paths:
        # Copier les masks combinés sans modification
        base_name = os.path.basename(path)
        out_path = os.path.join(output_dir, base_name)
        img = Image.open(path).convert('L')
        img = np.array(img).astype(np.float32)/255.0
        Image.fromarray((img*255).astype(np.uint8)).save(out_path)

    print(f"Normalization completed: {combined_count} images processed and saved to {output_dir}")