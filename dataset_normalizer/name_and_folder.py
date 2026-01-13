import os
import re
import shutil
from collections import defaultdict
from glob import glob

def name_in_folder(input_dir, output_dir):
    """
    Copy and normalize filenames for images and masks into output_dir.
    - input_dir: either [images_folder, masks_folder] or [single_folder_with_both]
    - output_dir: destination folder where files will be copied and renamed
    This function does NOT resize images; it only copies/renames files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Gather image and mask paths depending on input layout
    if len(input_dir) == 1:
        source = input_dir[0]
        all_files = sorted(glob(os.path.join(source, "*")))
        all_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

        images = []
        masks = []
        for p in all_files:
            if "_mask" not in os.path.basename(p).lower():
                images.append(p)
            else:
                masks.append(p)
    else:
        images = sorted(glob(os.path.join(input_dir[0], "*")))
        masks  = sorted(glob(os.path.join(input_dir[1], "*")))

    # Extract numeric id from filename (look for pattern like "(123)" or first sequence of digits)
    def extract_id(path):
        basename = os.path.basename(path)
        # Try to extract number from parentheses first: "benign (100)" -> 100
        m = re.search(r'\((\d+)\)', basename)
        if m:
            return int(m.group(1))
        # Fallback: first sequence of digits
        m = re.search(r'\d+', basename)
        return int(m.group(0)) if m else None

    # Group masks by the numeric id found in their filename
    masks_by_id = defaultdict(list)
    for mask_path in masks:
        idx = extract_id(mask_path)
        if idx is not None:
            masks_by_id[idx].append(mask_path)

    # Sort images by extracted id (numeric sort, not alphabetic)
    images_sorted = sorted(images, key=lambda p: extract_id(p) or 0)

    # Copy files into output_dir with consistent naming: img_000.png, img_000_mask_1.png, ...
    count, k = 0, 0
    for img_path in images_sorted:
        idx = extract_id(img_path)
        if idx is None:
            continue

        associated_masks = masks_by_id.get(idx, [])
        if not associated_masks:
            continue            # No GT → Ignore image

        # Keep original extension when copying
        img_ext = os.path.splitext(img_path)[1] or ".png"
        dest_img_name = f"img_{count:03d}{img_ext}"
        dest_img_path = os.path.join(output_dir, dest_img_name)
        shutil.copy2(img_path, dest_img_path)

        # Copy associated masks (if any) and enumerate them
        for i, mask_path in enumerate(associated_masks, start=1):
            mask_ext = os.path.splitext(mask_path)[1] or ".png"
            dest_mask_name = f"img_{count:03d}_mask_{i}{mask_ext}"
            dest_mask_path = os.path.join(output_dir, dest_mask_name)
            shutil.copy2(mask_path, dest_mask_path)
            k += 1

        count += 1
        
    print(f"Naming completed: {count + k} items copied to {output_dir}")