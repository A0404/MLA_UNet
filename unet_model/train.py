import os
from config import DEBUG_VIS, USE_DATA_AUG, USE_IGNORE_INDEX, USE_LOSS_POND
import torch
from torch.utils.data import DataLoader
from unet_model.model import UNet
from unet_model.debug import save_debug_image

def log_mask_stats(mask):
    unique, counts = torch.unique(mask, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}

# ========== EARLY STOPPING CLASS ==============================================
class EarlyStoppingTrain:
    def __init__(self, patience=50):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ========== SET SEED FUNCTION ================================================
def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== TRAINING FUNCTION ================================================
# --------- TRAINING FUNCTION FOR PAPER SETUP ------------------------------
def train_model_paper(
    model_dir,
    model_save, 
    train_ds,
    device,
    hyperparams=(1000, 1e-2, 0.99, 0.5, 1)
):
    """
    Train a UNet model on segmentation dataset.

    Args:
        root_dir: Path to directory containing training images and masks
        num_epochs: Number of training epochs (default 1000)
        learning_rate: Learning rate for optimizer (default 1e-2)
        momentum: Momentum for SGD optimizer (default 0.99)
        batch_size: Batch size for DataLoader (default 1)
        model_save_path: Path to save the trained model
        device: torch device (default uses CUDA if available)

    Returns:
        tuple: (model, train_losses, val_losses)
    """
    set_seed(42)

    # Unpack hyperparameters
    num_epochs, learning_rate, momentum,  dropout_rate, batch_size = hyperparams

    # Loaders
    #num_workers = max(1, min(8, os.cpu_count() // 2))
    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Create path
    save_path = os.path.join(model_dir, model_save)
    out_dir = "debug_images"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize model, optimizer, loss
    model = UNet(dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if USE_IGNORE_INDEX and not USE_LOSS_POND:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    elif not USE_IGNORE_INDEX and USE_LOSS_POND:
        class_weights = torch.tensor([1.0, 2.0]).to(device)         # Example weights for 2 classes
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif USE_LOSS_POND and USE_IGNORE_INDEX:
        class_weights = torch.tensor([1.0, 2.0]).to(device)         # Example weights for 2 classes
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    train_losses = []
    print("Starting training...\n")

    early_stop = EarlyStoppingTrain(patience=50)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        epoch_loss = 0
        for imgs, msks in train_loader:
            imgs = imgs.to(device)
            msks = msks.to(device).long()

            optimizer.zero_grad()                   # Zero gradients
            preds = model(imgs)                     # Forward pass

            # Debug: vérifier que preds ne contiennent pas d'explosions
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"[WARNING] NaN or Inf detected in preds at epoch {epoch}. Stopping training.")
                print("Preds stats:", preds.min().item(), preds.max().item(), preds.mean().item())
                return None, None  # stop training

            loss = criterion(preds, msks)           # Compute loss
            loss.backward()                         # Backward pass

            # Calcul de la norme totale des gradients
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # norme L2
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Clipping conditionnel
            if total_norm > 1.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                #print(f"[DEBUG] Gradient norm {total_norm:.3f} > 1.0, clipping applied")
            else:
                #print(f"[DEBUG] Gradient norm {total_norm:.3f}, no clipping needed")
                pass

            optimizer.step()                        # Update weights   
            epoch_loss += loss.item()               # Accumulate loss

        epoch_loss /= len(train_loader)             # Average loss
        train_losses.append(epoch_loss)

        # ---------------- DEBUG ----------------
        if epoch % 10 == 0:
            print("Unique mask values:", torch.unique(msks))
            cell_ratio = (msks == 1).sum().item() / msks.numel()
            print(f"[DEBUG] Cell ratio (batch_size=1): {cell_ratio:.4f}")
            print("[DEBUG] Mask stats:", log_mask_stats(msks))
            pred = torch.argmax(preds, dim=1)
            print("Unique predicted classes:", torch.unique(pred))
            if DEBUG_VIS:
                save_debug_image(imgs[0], msks[0], pred[0], os.path.join(out_dir, f"debug_DA_{USE_DATA_AUG}_INDEX_{USE_IGNORE_INDEX}_epoch_{epoch}.png"))
        # --------------------------------------

        early_stop.step(epoch_loss)
        if early_stop.stop:
            print("Early stopping triggered (train loss convergence).")
            break
        print(f"  Train Loss: {epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved as {model_save} in {model_dir}")

    return (model, train_losses)


# --------- TRAINING FUNCTION FOR PAPER SETUP ------------------------------
def train_model_study(
    model_dir,
    model_save,
    train_subset, 
    val_subset,
    hyperparams,
    device,
    resume=False,
):
    
    # Unpack hyperparameters
    num_epochs, learning_rate, momentum,  dropout_rate, batch_size = hyperparams

    # Loaders
    #num_workers = max(1, min(8, os.cpu_count() // 2))
    num_workers = 0
    train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Create path
    hyperparams_str = "_".join(str(h) for h in checkpoint['hyperparams'])
    save_path = os.path.join(model_dir, f"{hyperparams_str}_{model_save}")

    # Initialize model, optimizer, loss
    if resume and os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        num_epochs, learning_rate, momentum,  dropout_rate, batch_size = checkpoint['hyperparams']
    else:
        model = UNet(dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Training loop
    train_losses = []
    val_losses = []
    print("Starting training...\n")

    early_stop = EarlyStoppingTrain(patience=50)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        epoch_train_loss = 0
        for imgs, msks in train_loader:
            imgs = imgs.to(device)
            msks = msks.to(device).long()

            optimizer.zero_grad()                   # Zero gradients
            preds = model(imgs)                     # Forward pass
            loss = criterion(preds, msks)           # Compute loss
            loss.backward()                         # Backward pass
            optimizer.step()                        # Update weights   
            epoch_train_loss += loss.item()         # Accumulate loss

        epoch_train_loss /= len(train_loader)       # Average loss
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device)
                msks = msks.to(device).long()

                preds = model(imgs)                 # Forward pass    
                loss = criterion(preds, msks)       # Compute loss       
                epoch_val_loss += loss.item()       # Accumulate loss

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        early_stop.step(epoch_val_loss)
        if early_stop.stop:
            print("Early stopping triggered (train loss convergence).")
            break
        print(f"  Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # Create checkpoint dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparams': (num_epochs, learning_rate, momentum, dropout_rate, train_loader.batch_size)
    }    
    
    # Save model
    torch.save(checkpoint, save_path)
    print(f"\nModel saved as {hyperparams_str}_{model_save} in {model_dir}")

    return (model, train_losses, val_losses)