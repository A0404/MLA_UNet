import os
import re
import cv2
import numpy as np
from glob import glob

def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    """
    Robust image reader: tries cv2.imread first, 
    falls back to reading bytes + cv2.imdecode if needed.
    Works for TIFF, PNG, JPG, etc.
    """
    img = cv2.imread(path, flags)
    if img is None:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(arr, flags)
    return img

def write_image(path, img):
    """
    Writes an image as PNG (grayscale), ensures uint8 type.
    Creates directories if needed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if img.dtype != np.uint8:
        # Convert float images [0,1] to uint8 [0,255]
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    # Force PNG format
    _, buf = cv2.imencode(".png", img)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True

def mask_combiner(input_path, output_path, size=572, threshold=0.5):
    """
    Combine masks for each image and save as PNG.

    - Resize images to size_img (default 572x572)
    - Resize masks to size_mask (default 388x388)
    - Combine multiple masks per image by summing and thresholding
    - Save both images and masks as PNG in output_path
    """
    os.makedirs(output_path, exist_ok=True)
    all_files = sorted(glob(os.path.join(input_path, "*")))

    # Separate images from masks
    image_files = [p for p in all_files if "_mask" not in os.path.basename(p).lower()]
    mask_files  = [p for p in all_files if "_mask" in os.path.basename(p).lower()]

    def extract_index(path):
        """Extract numeric index from filename like 'img_000.png'"""
        m = re.search(r'img_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else None

    # Group masks by image index
    masks_by_index = {}
    for mpath in mask_files:
        idx = extract_index(mpath)
        if idx is not None:
            masks_by_index.setdefault(idx, []).append(mpath)

    combined_count = 0

    for img_path in image_files:
        idx = extract_index(img_path)
        if idx is None:
            continue

        # --- Read & resize image ---
        img = read_image(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        if img.shape[:2] != (size, size):
            interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_CUBIC
            img = cv2.resize(img, (size, size), interpolation=interp)
        # Save image as PNG
        out_img_path = os.path.join(output_path, os.path.splitext(os.path.basename(img_path))[0] + ".png")
        write_image(out_img_path, img)

        # --- Combine masks ---
        mask_list = masks_by_index.get(idx, [])
        if not mask_list:
            continue

        accumulated_mask = None
        for mask_path in mask_list:
            m = read_image(mask_path, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim > 2:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)     # Ensure grayscale
            if m.shape[:2] != (size, size):
                interp = cv2.INTER_AREA if max(m.shape[:2]) > size else cv2.INTER_CUBIC
                m = cv2.resize(m, (size, size), interpolation=interp)
            m = m.astype(np.float32) / 255.0    # Normalize to [0,1]
            # Accumulate masks by summing
            accumulated_mask = m if accumulated_mask is None else accumulated_mask + m

        if accumulated_mask is None:
            continue

        # Threshold to get binary mask
        final_mask = (accumulated_mask > threshold).astype(np.uint8) * 255
        out_mask_name = os.path.splitext(os.path.basename(img_path))[0] + "_combined_mask.png"
        write_image(os.path.join(output_path, out_mask_name), final_mask)
        combined_count += 1

    print(f"Combination completed: {combined_count} masks")