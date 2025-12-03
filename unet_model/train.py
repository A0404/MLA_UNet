import os
import torch
import matplotlib.pyplot as plt
from unet_model.dataset import train_loader
from unet_model.model import UNet


# ========== DEVICE ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== MODEL, OPTIMIZER, LOSS =============================================
model = UNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, momentum=0.99)
criterion = torch.nn.CrossEntropyLoss()

# ========== TRAINING LOOP =====================================================
num_epochs = 30
train_losses = []
print("Starting training...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        imgs, msks = batch
        imgs = imgs.to(device)
        msks = msks.to(device).long()

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, msks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Train Loss: {epoch_loss:.4f}")

# ========== SAVE MODEL =========================================================
torch.save(model.state_dict(), "unet_isbi.pth")
print("\nModel saved as unet_isbi.pth")

# ========== PLOT TRAINING CURVES ==============================================
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ========== VISUALIZE PREDICTIONS ==============================================
i = 0  # Index of the sample to visualize
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(imgs[i][0].cpu(), cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(msks[i][0].cpu(), cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(preds[i].argmax(dim=0).cpu().detach(), cmap="gray")

plt.show()