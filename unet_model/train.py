import os
from config import DEBUG_VIS, USE_DATA_AUG, USE_IGNORE_INDEX, USE_LOSS_POND
import torch
from torch.utils.data import DataLoader
from unet_model.model import UNet, init_weights_he
from unet_model.debug import save_debug_image

def log_mask_stats(mask):
    unique, counts = torch.unique(mask, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}

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
class EarlyStoppingTrain:
    def __init__(
        self,
        patience=80,
        min_delta=5e-4,
        ema_alpha=0.1,
        warmup_epochs=20
    ):
        """
        patience        : nb d'epochs sans amélioration tolérées
        min_delta       : amélioration minimale significative du Dice
        ema_alpha       : facteur de lissage EMA du Dice
        warmup_epochs   : epochs minimales avant autorisation d'arrêt
        """

        self.patience = patience
        self.min_delta = min_delta
        self.ema_alpha = ema_alpha
        self.warmup_epochs = warmup_epochs

        self.best_dice = -float("inf")
        self.best_recall_front = 0.0

        self.dice_ema = None
        self.counter = 0
        self.stop = False
        self.epoch = 0

    def step(self, dice_val, recall_front):
        self.epoch += 1

        # -------------------------
        # 1. Lissage EMA du Dice
        # -------------------------
        if self.dice_ema is None:
            self.dice_ema = dice_val
        else:
            self.dice_ema = (
                (1 - self.ema_alpha) * self.dice_ema
                + self.ema_alpha * dice_val
            )

        # -------------------------
        # 2. Détection d'amélioration
        # -------------------------
        dice_improved = self.dice_ema > self.best_dice + self.min_delta
        recall_improved = recall_front > self.best_recall_front + 1e-3

        improved = dice_improved or recall_improved

        if improved:
            if dice_improved:
                self.best_dice = self.dice_ema
            if recall_improved:
                self.best_recall_front = recall_front
            self.counter = 0
        else:
            self.counter += 1

        # -------------------------
        # 3. Sécurités critiques
        # -------------------------

        # a) Jamais d'arrêt pendant le warmup
        if self.epoch < self.warmup_epochs:
            self.counter = 0
            return

        # b) Jamais d'arrêt si frontières encore faibles
        if recall_front < 0.05:
            self.counter = 0
            return

        # -------------------------
        # 4. Décision finale
        # -------------------------
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

def load_checkpoint(path, model, optimizer, early_stop, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    early_stop.__dict__.update(checkpoint["early_stop"])

    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    print(f"[RESUME] Loaded checkpoint from {path} at epoch {start_epoch + 1}")

    return start_epoch, train_losses, val_losses

# ========== TRAINING FUNCTION ================================================
# --------- TRAINING FUNCTION FOR PAPER SETUP ------------------------------
def train_model_paper(
    model_dir,
    model_save, 
    train_ds,
    val_ds,
    device,
    hyperparams=(1000, 5e-3, 0.99, 0.5, 1),
    resume = False
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

    # Unpack hyperparameters
    num_epochs, learning_rate, momentum,  dropout_rate, batch_size = hyperparams

    # Create path
    save_path = os.path.join(model_dir, model_save)
    out_dir = "debug_images"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize model, optimizer, loss
    model = UNet(dropout_rate=dropout_rate).to(device)
    model.apply(init_weights_he)

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: std={param.std().item():.4f}, mean={param.mean().item():.4f}")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = UNetWeightedCELoss()

    early_stop = EarlyStoppingTrain(
        patience=80,
        min_delta=5e-4,
        ema_alpha=0.1,
        warmup_epochs=20
    )

    # Initialize variables
    train_losses = []
    val_losses = []
    start_epoch = 0

    if resume and os.path.exists(save_path):
        start_epoch, train_losses, val_losses = load_checkpoint(
            save_path,
            model,
            optimizer,
            early_stop,
            device
        )

    # Loaders
    #num_workers = max(1, min(8, os.cpu_count() // 2))
    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # --- Après avoir créé le DataLoader ---
    for imgs, msks, wmaps in train_loader:
        print("Mask unique values:", torch.unique(msks))        # doit être [0,1]
        print("Weight map stats: min", wmaps.min().item(),
          "max", wmaps.max().item(),
          "mean", wmaps.mean().item())                     # min/max doivent varier autour de 1
        break  # juste le premier batch pour debug

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

            """
            cell_ratio = (msks == 1).float().mean().item()
            print(f"[DEBUG] Cell ratio in batch: {cell_ratio:.4f}")"""

            wmaps = wmaps.to(device).float()

            # --- SANITY CHECK INPUTS ---
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                raise RuntimeError("NaN/Inf detected in INPUT IMAGES")

            if torch.isnan(msks).any() or torch.isinf(msks).any():
                raise RuntimeError("NaN/Inf detected in MASKS")

            if torch.isnan(wmaps).any() or torch.isinf(wmaps).any():
                raise RuntimeError("NaN/Inf detected in WEIGHT MAPS")

            if wmaps.min() <= 0:
                print("[WARNING] weight map has non-positive values:", wmaps.min().item())

            #print("image :",imgs,"mask", msks, "weight_map", wmaps)
    
            optimizer.zero_grad()                   # Zero gradients
            preds = model(imgs / 255.0)             # Forward pass
            #print("preds:", preds)

            """
            # Inspecter les activations
            print("Logits stats: min", preds.min().item(),
                "max", preds.max().item(),
                "mean", preds.mean().item())"""

            # Debug: vérifier que preds ne contiennent pas d'explosions
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"[WARNING] NaN or Inf detected in preds at epoch {epoch + 1}. Stopping training.")
                print("Preds stats:", preds.min().item(), preds.max().item(), preds.mean().item())
                return None, None  # stop training

            loss = criterion(preds, msks, wmaps)           # Compute loss
            loss.backward()                         # Backward pass

            """
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
            """
                
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

                # --- SANITY CHECK INPUTS ---
                if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                    raise RuntimeError("NaN/Inf detected in INPUT IMAGES")

                if torch.isnan(msks).any() or torch.isinf(msks).any():
                    raise RuntimeError("NaN/Inf detected in MASKS")

                if torch.isnan(wmaps).any() or torch.isinf(wmaps).any():
                    raise RuntimeError("NaN/Inf detected in WEIGHT MAPS")

                if wmaps.min() <= 0:
                    print("[WARNING] weight map has non-positive values:", wmaps.min().item())

                preds = model(imgs / 255.0)             # Forward pass
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
        if epoch % 1 == 0:
            #print("Unique mask values:", torch.unique(msks))
            cell_ratio = (msks == 1).sum().item() / msks.numel()
            #print(f"[DEBUG] Cell ratio (batch_size=1): {cell_ratio:.4f}")
            #print("[DEBUG] Mask stats:", log_mask_stats(msks))

            # Probabilités brutes
            with torch.no_grad():
                probs = torch.softmax(preds, dim=1)  # shape [B, C, H, W]
                # Exemple pour le premier batch
                probs_img = probs[0].detach().cpu().numpy()  # shape [C, H, W]
                #print("Probs shape:", probs_img.shape)
                print("Probs min/max per class:", probs_img.min(axis=(1,2)), probs_img.max(axis=(1,2)))

            pred = torch.argmax(preds, dim=1)
            print("Unique predicted classes:", torch.unique(pred))
            
            pred_image = pred.float() * 255
            if torch.mean(pred_image) < 128:  # plus de pixels blancs que noirs
                pred_image = 255 - pred_image
                
            if DEBUG_VIS:
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
        early_stop.step(dice_mean, recall_front_mean)
        if early_stop.stop:
            print(
                f"[EARLY STOP] Convergence atteinte | "
                f"Best Dice: {early_stop.best_dice:.4f}"
            )
            break

        print(f"  Train Loss: {epoch_loss:.4f}")

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