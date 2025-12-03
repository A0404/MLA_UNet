import os
from glob import glob
import shutil
import re

# --- CONFIG ---
images_dir = r"C:\Users\adrie\Documents\5A\MLA\bdd\isbi-datasets\data\images"
masks_dir  = r"C:\Users\adrie\Documents\5A\MLA\bdd\isbi-datasets\data\labels"
output_dir = r"C:\Users\adrie\Documents\5A\MLA\bdd\isbi-datasets\formed"

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get all image and mask files
images = sorted(glob(os.path.join(images_dir, "*.*")))
masks  = sorted(glob(os.path.join(masks_dir, "*.*")))

# Function to extract a number from a filename
def extract_number(filename):
    # Search for the first number in the filename
    match = re.search(r'\d+', os.path.basename(filename))
    if match:
        return int(match.group())
    else:
        return None

# Create a dictionary for masks: number -> path
mask_dict = {}
for m in masks:
    num = extract_number(m)
    if num is not None:
        mask_dict[num] = m

# Copy and rename files with a uniform naming convention
counter = 0
for img_path in images:
    num = extract_number(img_path)
    if num is None:
        continue

    if num not in mask_dict:
        print(f"No mask found for image {img_path}")
        continue

    mask_path = mask_dict[num]

    # New uniform names
    img_new_name  = f"img_{counter:03d}.png"
    mask_new_name = f"img_{counter:03d}_mask.png"

    # Destination paths
    img_dst  = os.path.join(output_dir, img_new_name)
    mask_dst = os.path.join(output_dir, mask_new_name)

    # Copy files to the output directory
    shutil.copy2(img_path, img_dst)
    shutil.copy2(mask_path, mask_dst)

    counter += 1

print(f"Normalization completed: {counter} images/masks copied to {output_dir}")