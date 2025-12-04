import os
import re
import cv2
import numpy as np
from glob import glob

def _cv_read_any(path, flags=cv2.IMREAD_UNCHANGED):
    """
    Robust image reader:
    - try cv2.imread
    - fallback to reading bytes + cv2.imdecode for paths that cv2 can't handle directly
    """
    img = cv2.imread(path, flags)
    if img is None:
        try:
            with open(path, "rb") as f:
                data = f.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, flags)
        except Exception:
            return None
    return img

def _cv_write_any(path, img, params=None):
    """
    Robust image writer using cv2.imencode + Python open to support unicode/odd paths.
    Forces PNG encoding and ensures uint8 dtype for encoding.
    Returns True on success, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Ensure image is uint8 for imencode
        if img.dtype != np.uint8:
            if img.dtype in (np.float32, np.float64):
                img_to_encode = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                img_to_encode = img.astype(np.uint8)
        else:
            img_to_encode = img

        ok, buf = cv2.imencode(".png", img_to_encode, params or [])
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False

def mask_combiner(input_path, output_path, size=572, threshold=0.5):
    """
    Read files from input_path (produced by normalizer), optionally resize images/masks,
    combine all masks corresponding to each image and write results into output_path.

    Combines masks by accumulating them then thresholding (like image_mask_combiner_1.py).

    Output files:
     - overwritten image: img_XXX.ext  (grayscale, resized if needed)
     - combined mask:   img_XXX_combined_mask.png
    """
    os.makedirs(output_path, exist_ok=True)

    all_files = sorted(glob(os.path.join(input_path, "*")))
    image_files = [p for p in all_files if "_mask" not in os.path.basename(p).lower()]
    mask_files  = [p for p in all_files if "_mask" in os.path.basename(p).lower()]

    # Extract index from filename like 'img_000' -> 0
    def extract_index(path):
        m = re.search(r'img_(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else None

    # Group masks by image index
    masks_by_index = {}
    for mpath in mask_files:
        idx = extract_index(mpath)
        if idx is None:
            continue
        masks_by_index.setdefault(idx, []).append(mpath)

    combined_count = 0

    for img_path in image_files:
        idx = extract_index(img_path)
        if idx is None:
            continue

        # Read image (from temp folder)
        img = _cv_read_any(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Resize image if needed
        h, w = img.shape[:2]
        if (h, w) != (size, size):
            interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
            img = cv2.resize(img, (size, size), interpolation=interp)

        # Ensure grayscale before writing back to output_path
        if img.ndim > 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        _cv_write_any(os.path.join(output_path, os.path.basename(img_path)), img_gray.astype(np.uint8))

        # Combine masks for this image index (logic from image_mask_combiner_1.py)
        mask_list = masks_by_index.get(idx, [])
        if not mask_list:
            continue

        accumulated_mask = None
        for mask_path in mask_list:
            mimg = _cv_read_any(mask_path, cv2.IMREAD_UNCHANGED)
            if mimg is None:
                continue

            mh, mw = mimg.shape[:2]
            if (mh, mw) != (size, size):
                interp = cv2.INTER_AREA if max(mh, mw) > size else cv2.INTER_CUBIC
                mimg = cv2.resize(mimg, (size, size), interpolation=interp)

            if mimg.ndim > 2:
                mimg = cv2.cvtColor(mimg, cv2.COLOR_BGR2GRAY)

            m = mimg.astype(np.float32) / 255.0
            # Accumulate: if first mask, initialize; else add
            if accumulated_mask is None:
                accumulated_mask = m
            else:
                accumulated_mask += m

        if accumulated_mask is None:
            continue

        # Threshold after accumulation (like image_mask_combiner_1.py)
        final_mask = np.where(accumulated_mask > threshold, 1.0, 0.0)
        final_mask = (final_mask * 255).astype(np.uint8)

        out_mask_name = os.path.splitext(os.path.basename(img_path))[0] + "_combined_mask.png"
        out_mask_path = os.path.join(output_path, out_mask_name)
        if _cv_write_any(out_mask_path, final_mask):
            combined_count += 1

    # Summary
    print(f"Combinaison completed: {combined_count} combined masks")