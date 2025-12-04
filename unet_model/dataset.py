import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


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
        base, _ = os.path.splitext(img_path)
        mask_path = base + "_combined_mask.png"

        # Retrieving images and masks
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Transformations to tensors and mask binarization
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()

        return (image, mask)
    

# ========== DATASET & DATALOADER CREATION =====================================
def dataset_loaders(root_dir, ratio=0.8, batch_size=4):
    # Split train/val
    all_pngs = sorted(glob(os.path.join(root_dir, "*.png")))
    images = [p for p in all_pngs if not p.endswith("_combined_mask.png")]

    n = int(ratio * len(images))
    train_files = images[:n]
    val_files = images[n:]

    # Datasets
    train_ds = SegmentationDataset(train_files)
    val_ds = SegmentationDataset(val_files)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)
    return (train_loader, val_loader)