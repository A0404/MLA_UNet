import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from unet_model.model import UNet, init_weights_he
from config import DEBUG_VIS

# ========== DEBUG IMAGE ======================================
def save_debug_image(img, mask, pred, out_path):
    """
    img: Tensor [1,H,W] or [H,W]
    mask: Tensor [H,W]
    pred: Tensor [H,W]
    """

    img = img.squeeze().detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Image")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("GT Mask")
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title("Prediction")

    for a in ax:
        a.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ========== WEIGHTED CROSS ENTROPY LOSS ======================================
class UNetWeightedCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target, weight_map):
        ce_loss = self.ce(logits, target)   # (B,H,W)
        loss = ce_loss * weight_map
        return loss.mean()


# ========== EARLY STOPPING CLASS ==============================================
class EarlyStoppingFrontier:
    def __init__(self, patience=20, min_delta=1e-4, warmup_epochs=30, min_frontier_frac=0.08):
        """
        Early stopping based on BORDER quality (EM/ISBI).

        Patience:           Epochs with no improvement tolerated.
        min_delta:          Minimum significant improvement in recall.
        warmup_epochs:      Minimum epochs before possible stopping.
        min_frontier_frac:  Minimum fraction of membrane pixels accepted.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.min_frontier_frac = min_frontier_frac

        self.best_val_loss = float("inf")
        self.best_recall = min_delta
        self.counter = 0
        self.stop = False
        self.epoch = 0

    def step(self, recall_front, frontier_frac, val_loss):
        self.epoch += 1

        print(
            f"[BEST] "
            f"Best Loss: {self.best_val_loss:.4f} | "
            f"Best Recall(front): {self.best_recall:.3f} | "
            f"Best Loss/Recall: {self.best_val_loss/self.best_recall:.3f}"
        )

        # 1. Warm-up: never stop prematurely
        if self.epoch <= self.warmup_epochs:
            if ((val_loss/recall_front) < (self.best_val_loss/self.best_recall)) or (
                val_loss < (self.best_val_loss + self.min_delta)):
                self.best_recall = recall_front
                self.best_val_loss = val_loss
            return False

        # 2. Structural consistency check
        if frontier_frac < self.min_frontier_frac:
            # Degenerate model → Improvement is not even considered
            self.counter += 1
        else:
            # 3. Significant improvement?
            if ((val_loss/recall_front) < (self.best_val_loss/self.best_recall)) or (
                val_loss < (self.best_val_loss + self.min_delta)):
                self.best_recall = recall_front
                self.best_val_loss = val_loss
                self.stop = False
                self.counter = 0
            else:
                self.counter += 1

        # 4. Stopping decision
        if self.counter >= self.patience:
            self.stop = True

        return self.stop


# ========== SET SEED FUNCTION ================================================
def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== METRICS ================================================
def dice_coefficient(pred, target, eps=1e-6):
    """
    Dice for binary segmentation (class 1)
    pred, target: (H, W) in {0,1}
    """
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def recall_frontier(pred, target, eps=1e-6):
    """
    Recall on the boundary class (assumed = 0)
    """
    tp = ((pred == 0) & (target == 0)).sum().float()
    fn = ((pred == 1) & (target == 0)).sum().float()
    return tp / (tp + fn + eps)


# ========== CHECKPOINT ================================================
def save_checkpoint(path, model, optimizer, epoch, early_stop, train_losses, val_losses, recall_liste, val_recall, hyperparams):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "early_stop": early_stop.__dict__,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "recall_liste": recall_liste,
        "val_recall": val_recall,
        "hyperparams": hyperparams
    }
    torch.save(checkpoint, path)
    
def load_or_initialize(save_path, device, hyperparams, resume=False):
    # Load hyperparams if resume=True and file exists
    if resume and os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        hyperparams = ckpt["hyperparams"]

    num_epochs, lr, momentum, dropout_rate, batch_size = hyperparams
    
    # Initialize all components
    model = UNet(dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    start_epoch = 0
    train_losses, val_losses, recall_liste, val_recall = [], [], [], []

    early_stop = EarlyStoppingFrontier()

    # Load checkpoint if resume=True
    if resume and os.path.exists(save_path):
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        early_stop.__dict__.update(ckpt["early_stop"])

        start_epoch = ckpt["epoch"] + 1
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        recall_liste = ckpt["recall_liste"]
        val_recall = ckpt["val_recall"]

        print(f"[RESUME] Loaded checkpoint at epoch {start_epoch}")
    # Otherwise, initialize weights
    else:
        model.apply(init_weights_he)

    return (model, optimizer, early_stop, start_epoch, num_epochs, batch_size, train_losses, val_losses, recall_liste, val_recall)


# ========== TRAINING FUNCTION ================================================
def train_model(model_dir, model_save,  train_ds, val_ds, device, resume = False, hyperparams=(1000, 3e-4, 0.99, 0.05, 1)):
    """
    Train a UNet model on segmentation dataset.

    Args:
        root_dir: Path to directory containing training images and masks
        num_epochs: Number of training epochs (default 1000)
        learning_rate: Learning rate for optimizer (default 3e-4)
        momentum: Momentum for SGD optimizer (default 0.99)
        batch_size: Batch size for DataLoader (default 1)
        model_save_path: Path to save the trained model
        device: torch device (default uses CUDA if available)

    Returns:
        tuple: (model, train_losses, val_losses, front_liste, val_front)
    """
    set_seed(42)

    # Create path
    save_path = os.path.join(model_dir, model_save)

    # Debug images directory
    out_dir = "debug_images"
    image_path = os.path.join(out_dir, model_save)
    os.makedirs(image_path, exist_ok=True)

    # Load or initialize model
    (model, optimizer, early_stop, start_epoch, num_epochs, batch_size, train_losses, val_losses, recall_liste, val_recall 
    ) = load_or_initialize(save_path=save_path, device=device, hyperparams=hyperparams, resume=resume)

    # Loaders
    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --------------------------------------------------------
    # Loss function
    criterion = UNetWeightedCELoss()

    # Training loop
    print("Starting training...")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        epoch_loss = 0
        for imgs, msks, wmaps in train_loader:
            imgs = imgs.to(device)
            msks = msks.to(device).long()
            wmaps = wmaps.to(device).float()
    
            optimizer.zero_grad()                   # Zero gradients
            preds = model(imgs)             # Forward pass

            loss = criterion(preds, msks, wmaps)           # Compute loss
            loss.backward()                         # Backward pass
                
            optimizer.step()                        # Update weights   
            epoch_loss += loss.item()               # Accumulate loss

        epoch_loss /= len(train_loader)             # Average loss
        train_losses.append(epoch_loss)

        # -----------------
        # "Validation" phase (modele évalué sur le même dataset)
        # -----------------
        model.eval()
        val_loss = 0
        dice_sum = 0.0
        recall_front_sum = 0.0
        frontier_frac_sum = 0.0

        with torch.no_grad():
            for imgs, msks, wmaps in val_loader:  # même loader, juste eval
                imgs = imgs.to(device)
                msks = msks.to(device).long()
                wmaps = wmaps.to(device).float()

                preds = model(imgs)             # Forward pass
                loss = criterion(preds, msks, wmaps)
                val_loss += loss.item()

                # --- Predictions binaires ---
                pred_classes = torch.argmax(preds, dim=1)

                # --- Métriques ---
                dice = dice_coefficient(pred_classes, msks)
                recall_f = recall_frontier(pred_classes, msks)
                frontier_frac = (pred_classes == 0).float().mean()

                dice_sum += dice.item()
                recall_front_sum += recall_f.item()
                frontier_frac_sum += frontier_frac.item()
    
        val_loss /= len(val_loader)
        dice_mean = dice_sum / len(val_loader)
        recall_front_mean = recall_front_sum / len(val_loader)
        frontier_frac_mean = frontier_frac_sum / len(val_loader)

        val_losses.append(val_loss)
        recall_liste.append(recall_front_mean)
        val_recall.append(val_loss/recall_front_mean)

        # ---------------- DEBUG ----------------
        if DEBUG_VIS and (epoch+1)%10 == 0:
            # Probabilités brutes
            with torch.no_grad():
                probs = torch.softmax(preds, dim=1)  # shape [B, C, H, W]
                MEMBRANE_CLASS = 1                   # or 0 depending on your encoding
                membrane_prob = probs[:, MEMBRANE_CLASS]
    
                # Exemple pour le premier batch
                #probs_img = probs[0].detach().cpu().numpy()  # shape [C, H, W]
                #print("Probs min/max per class:", probs_img.min(axis=(1,2)), probs_img.max(axis=(1,2)))

            #pred = torch.argmax(preds, dim=1)
            #print("Unique predicted classes:", torch.unique(pred))
            
            pred_image = (membrane_prob > 0.5).float() * 255
            
            save_debug_image(imgs[0], msks[0], pred_image[0], os.path.join(image_path, f"{model_save}_epoch_{epoch + 1}.png"))
        # --------------------------------------

        # ---------------- CHECKPOINTING ----------------
        save_checkpoint(save_path, model, optimizer, epoch, early_stop, train_losses, val_losses, recall_liste, val_recall, hyperparams)
        print(f"Model saved as {model_save} in {model_dir} with checkpoint at epoch {epoch + 1}")

        # ---------------- METRICS ----------------
        print(f"[TRAIN] Loss: {epoch_loss:.4f}")
        print(
            f"[VAL] "
            f"Loss: {val_loss:.4f} | "
            f"Dice(cell): {dice_mean:.3f} | "
            f"Recall(front): {recall_front_mean:.3f} | "
            f"Frontier frac: {frontier_frac_mean:.3f} | "
            f"Loss/Recall: {val_loss/recall_front_mean:.3f}"
        )

        # ---------------- EARLY STOPPING ----------------
        if early_stop.step(recall_front_mean, frontier_frac_mean, val_loss):
            print("Early stopping triggered (loss and loss/frontier based).")
            break

    # Final save
    save_checkpoint(save_path, model, optimizer, epoch, early_stop, train_losses, val_losses, recall_liste, val_recall, hyperparams)
    print(f"\nModel saved as {model_save} in {model_dir} with checkpoint at epoch {epoch + 1}\n")

    return (model, train_losses, val_losses, recall_liste, val_recall)