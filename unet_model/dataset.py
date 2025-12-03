import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ========== PATHS =============================================================
root_dir = r"C:\Users\adrie\Documents\5A\MLA\bdd\isbi-datasets\formed"

# ========== DATASET DEFINITION ================================================
class SegmentationDataset(Dataset):
    def __init__(self, file_list):
        # file_list : List of paths to the images (not the masks)
        self.images = file_list
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieving paths
        img_path = self.images[idx]
        mask_path = img_path.replace(".png", "_combined_mask.png")

        # Retrieving images and masks
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Transformations to tensors and mask binarization
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()

        return (image, mask)
    

# ========== DATASET & DATALOADER CREATION =====================================
# Split train/val
images = sorted(glob(os.path.join(root_dir, "*[!_mask].png")))
n = int(0.8 * len(images))
train_files = images[:n]
val_files = images[n:]

# Datasets
train_ds = SegmentationDataset(train_files)
val_ds = SegmentationDataset(val_files)

# Loaders
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)