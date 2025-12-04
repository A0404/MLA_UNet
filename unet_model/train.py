import os
import torch
import matplotlib.pyplot as plt
from unet_model.model import UNet
from unet_model.dataset import dataset_loaders


def train_model(
    root_dir,
    num_epochs,
    learning_rate,
    momentum,
    batch_size,
    model_save_path,
    device
):
    """
    Train a UNet model on segmentation dataset.

    Args:
        root_dir: Path to directory containing training images and masks
        num_epochs: Number of training epochs (default 30)
        learning_rate: Learning rate for optimizer (default 3e-3)
        momentum: Momentum for SGD optimizer (default 0.99)
        batch_size: Batch size for DataLoader (default 4)
        model_save_path: Path to save the trained model (default "unet_isbi.pth")
        device: torch device (default uses CUDA if available)

    Returns:
        tuple: (model, train_losses, val_losses)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader = dataset_loaders(
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle_train=True
    )

    # Initialize model, optimizer, loss
    model = UNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    train_losses = []
    val_losses = []
    print("Starting training...\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        epoch_train_loss = 0
        for imgs, msks in train_loader:
            imgs = imgs.to(device)
            msks = msks.to(device).long()

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, msks)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device)
                msks = msks.to(device).long()

                preds = model(imgs)
                loss = criterion(preds, msks)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"  Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved as {model_save_path}")

    return (model, train_losses, val_losses)


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_predictions(model, imgs, msks, device, num_samples=3):
    """Visualize model predictions."""
    model.eval()
    with torch.no_grad():
        preds = model(imgs.to(device))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(min(num_samples, len(imgs))):
        axes[i, 0].imshow(imgs[i][0].cpu(), cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(msks[i][0].cpu(), cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i].argmax(dim=0).cpu().detach(), cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()