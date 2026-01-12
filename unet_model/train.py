import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from unet_model.model import UNet, init_weights_he
from config import DEBUG_VIS


# ========== LOG MASK STATS ======================================
def log_mask_stats(mask):
    unique, counts = torch.unique(mask, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


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
        self.ce = torch.nn.CrossEntropyLoss(
            reduction="none"
        )

    def forward(self, logits, target, weight_map):
        ce_loss = self.ce(logits, target)   # (B,H,W)
        loss = ce_loss * weight_map
        return loss.mean()


# ========== EARLY STOPPING CLASS ==============================================
class EarlyStoppingValLoss:
    def __init__(
        self,
        patience=20,
        min_delta=1e-4,
        warmup_epochs=20
    ):
        """
        Early stopping basé UNIQUEMENT sur la validation loss.

        patience        : nombre d'epochs sans amélioration tolérées
        min_delta       : amélioration minimale significative de la val loss
        warmup_epochs   : epochs minimales avant autorisation d'arrêt
        """

        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs

        self.best_val_loss = float("inf")
        self.counter = 0
        self.stop = False
        self.epoch = 0

    def step(self, val_loss):
        self.epoch += 1

        # 1. Warmup : on n'arrête jamais trop tôt
        if self.epoch <= self.warmup_epochs:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            return

        # 2. Amélioration significative ?
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        # 3. Décision d'arrêt
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


# ========== METRICS ================================================
def dice_coefficient(pred, target, eps=1e-6):
    """
    Dice pour segmentation binaire (classe 1)
    pred, target : (H, W) en {0,1}
    """
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def recall_frontier(pred, target, eps=1e-6):
    """
    Recall sur la classe frontière (supposée = 0)
    """
    tp = ((pred == 0) & (target == 0)).sum().float()
    fn = ((pred == 1) & (target == 0)).sum().float()
    return tp / (tp + fn + eps)


# ========== CHECKPOINT ================================================
def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    early_stop,
    train_losses,
    val_losses,
    hyperparams
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "early_stop": early_stop.__dict__,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "hyperparams": hyperparams
    }
    torch.save(checkpoint, path)
    
def load_or_initialize(
    save_path,
    device,
    hyperparams,
    resume=False
):
    # Load hyperparams if resume=True and file exists
    if resume and os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        hyperparams = ckpt["hyperparams"]

    num_epochs, lr, momentum, dropout_rate, batch_size = hyperparams
    
    # Initialize all components
    model = UNet(dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    start_epoch = 0
    train_losses, val_losses = [], []

    early_stop = EarlyStoppingValLoss()

    # Load checkpoint if resume=True
    if resume and os.path.exists(save_path):
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        early_stop.__dict__.update(ckpt["early_stop"])

        start_epoch = ckpt["epoch"] + 1
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]

        print(f"[RESUME] Loaded checkpoint at epoch {start_epoch}")
    # Otherwise, initialize weights
    else:
        model.apply(init_weights_he)

    return (
        model,
        optimizer,
        early_stop,
        start_epoch,
        num_epochs,
        batch_size,
        train_losses,
        val_losses
    )


# ========== TRAINING FUNCTION ================================================
def train_model(
    model_dir,
    model_save, 
    train_ds,
    val_ds,
    device,
    resume = False,
    hyperparams=(1000, 1e-3, 0.99, 0.2, 1)
):
    """
    Train a UNet model on segmentation dataset.

    Args:
        root_dir: Path to directory containing training images and masks
        num_epochs: Number of training epochs (default 1000)
        learning_rate: Learning rate for optimizer (default 5e-3)
        momentum: Momentum for SGD optimizer (default 0.99)
        batch_size: Batch size for DataLoader (default 1)
        model_save_path: Path to save the trained model
        device: torch device (default uses CUDA if available)

    Returns:
        tuple: (model, train_losses, val_losses)
    """
    set_seed(42)

    # Create path
    save_path = os.path.join(model_dir, model_save)
    out_dir = "debug_images"
    os.makedirs(out_dir, exist_ok=True)

    # Load or initialize model
    (model, optimizer, early_stop, start_epoch, num_epochs, batch_size, train_losses, val_losses
    ) = load_or_initialize(save_path=save_path, device=device, hyperparams=hyperparams, resume=resume)

    # Loaders
    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --------------------------------------------------------
    # Loss function
    criterion = UNetWeightedCELoss()

    # Training loop
    print("Starting training...\n")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

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

        print(f"[TRAIN] Loss: {epoch_loss:.4f}")

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
        print(
            f"[VAL] "
            f"Loss: {val_loss:.4f} | "
            f"Dice(cell): {dice_mean:.3f} | "
            f"Recall(front): {recall_front_mean:.3f} | "
            f"Frontier frac: {frontier_frac_mean:.3f}"
        )

        # ---------------- DEBUG ----------------
        if DEBUG_VIS:
            # Probabilités brutes
            with torch.no_grad():
                probs = torch.softmax(preds, dim=1)  # shape [B, C, H, W]
                # Exemple pour le premier batch
                probs_img = probs[0].detach().cpu().numpy()  # shape [C, H, W]
                print("Probs min/max per class:", probs_img.min(axis=(1,2)), probs_img.max(axis=(1,2)))

            pred = torch.argmax(preds, dim=1)
            print("Unique predicted classes:", torch.unique(pred))
            
            pred_image = pred.float() * 255
            if torch.mean(pred_image) < 128:  # plus de pixels blancs que noirs
                pred_image = 255 - pred_image
            
            save_debug_image(imgs[0], msks[0], pred_image[0], os.path.join(out_dir, f"{model_save}_epoch_{epoch + 1}.png"))
        # --------------------------------------

        # ---------------- CHECKPOINTING ----------------
        save_checkpoint(
            save_path,
            model,
            optimizer,
            epoch,
            early_stop,
            train_losses,
            val_losses,
            hyperparams
        )
        print(f"\nModel saved as {model_save} in {model_dir} with checkpoint at epoch {epoch + 1}\n")

        # ---------------- EARLY STOPPING ----------------
        early_stop.step(val_loss)
        if early_stop.stop:
            print(
                f"[EARLY STOP] Validation loss n'améliore plus "
                f"(best = {early_stop.best_val_loss:.4f})"
            )
            break

    # Final save
    save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch,
                early_stop,
                train_losses,
                val_losses,
                hyperparams
            )
    print(f"\nModel saved as {model_save} in {model_dir} with checkpoint at epoch {epoch + 1}\n")

    return (model, train_losses, val_losses)