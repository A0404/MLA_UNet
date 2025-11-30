# -*- coding: utf-8 -*-
"""
U-Net training on BUSI cleaned_data (malignant)
@author: Aymen
"""

# ========== IMPORTS ===========================================================
import os
from glob import glob
import torch
import monai
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    Resized, EnsureTyped, Compose, Lambdad
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt

# ========== PATHS =============================================================
root_dir = r"C:\MesEtudes\SAR\MLA\cleaned_data\malignant"

# ========== DEVICE ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== READ DATASET ======================================================
images = sorted(glob(os.path.join(root_dir, "*[!_mask].png")))

data = []
for img in images:
    base = img.replace(".png", "")
    mask1 = base + "_mask.png"
    mask2 = base + "_combined_mask.png"

    if os.path.exists(mask1):
        data.append({"image": img, "mask": mask1})
    elif os.path.exists(mask2):
        data.append({"image": img, "mask": mask2})
    else:
        print("⚠️ Missing mask for:", img)

print("Total valid image/mask pairs:", len(data))

# Train/validation split
n = int(0.8 * len(data))
train_files = data[:n]
val_files = data[n:]

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# ========== TRANSFORMS ========================================================
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),

    # Convert image RGB → grayscale
    Lambdad(keys="image", func=lambda x: x[0:1]),

    # Convert mask to binary (values > 0 → 1)
    Lambdad(keys="mask", func=lambda x: (x > 0).float()),

    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "mask"], spatial_size=(256, 256)),
    EnsureTyped(keys=["image", "mask"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    Lambdad(keys="image", func=lambda x: x[0:1]),
    Lambdad(keys="mask", func=lambda x: (x > 0).float()),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "mask"], spatial_size=(256, 256)),
    EnsureTyped(keys=["image", "mask"]),
])

# ========== DATASETS & LOADERS ===============================================
train_ds = Dataset(train_files, train_transforms)
val_ds = Dataset(val_files, val_transforms)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

# ========== MODEL =============================================================
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,  # binary segmentation
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# ========== TRAINING LOOP =====================================================
num_epochs = 30
train_losses, val_dices = [], []

print("Starting training...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        imgs = batch["image"].to(device)
        msks = batch["mask"].to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, msks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Train Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        dices = []
        for batch in val_loader:
            vimg = batch["image"].to(device)
            vmsk = batch["mask"].to(device)
            vpred = model(vimg)
            d = dice_metric(y_pred=vpred, y=vmsk)

            # FIX: convert batch of dice values → scalar
            dices.append(d.mean().item())

        mean_dice = sum(dices) / len(dices)
        val_dices.append(mean_dice)
        print(f"Validation Dice: {mean_dice:.4f}")

# ========== SAVE MODEL =========================================================
torch.save(model.state_dict(), "unet_busi_cleaned.pth")
print("\nModel saved as unet_busi_cleaned.pth")

# ========== PLOT TRAINING CURVES ==============================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(val_dices)
plt.title("Validation Dice")

plt.show()

# ========== SHOW SAMPLE PREDICTION ============================================
model.eval()
batch = next(iter(val_loader))
img = batch["image"].to(device)
msk = batch["mask"].to(device)

with torch.no_grad():
    pred = model(img)
    pred = torch.argmax(pred, dim=1).cpu()

i = 0
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(img[i][0].cpu(), cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(msk[i][0].cpu())

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred[i])

plt.show()
