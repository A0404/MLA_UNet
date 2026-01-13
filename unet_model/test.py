import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet_model.model import UNet
from scipy.ndimage import rotate, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from unet_model.metrics import pixel_error, rand_error, warping_error_normalized, dice, iou_score

# ========== CHECKPOINT ================================================
def load_checkpoint(save_path, device):
    """ Load model and batch size from checkpoint. """
    ckpt = torch.load(save_path, weights_only=True)
    hyperparams = ckpt["hyperparams"]
    _, _, _, dropout_rate, batch_size = hyperparams

    model = UNet(dropout_rate=dropout_rate).to(device)
    model.load_state_dict(ckpt["model_state"])

    return (model, batch_size)

def safe_mean(lst):
    lst = [x for x in lst if not np.isnan(x)]
    return np.mean(lst) if len(lst) > 0 else float('nan')

# ====== PLOT IMAGE, MASK, PRED =========================================
def plot_image(img, mask_np, pred):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img.cpu().numpy()[0, 0], cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------------
#   TESTS FUNCTIONS
# -------------------------------
def test_em_unet(save_path, test_ds, device, thresholds=np.linspace(0.2, 0.29, 10), rot_angles = [0, 45, 90, 135, 180, 225, 270], num_samples_to_show=2):
    # Unpack hyperparameters
    model, batch_size = load_checkpoint(save_path, device)
    model.eval()

    # Loaders
    num_workers = 0
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    metrics = {
        t: {"pixel": [], "rand": [], "warp": [], "prec": [], "rec": []}
        for t in thresholds
    }

    k = 0
    with torch.no_grad():
        for i, (img, mask, _) in enumerate(test_loader):
            img = img.to(device)
            mask_np = mask.cpu().numpy()[0]  # ground truth pour cette image

            # 7 rotations
            prob_maps = []
            for angle in rot_angles:
                rotated_img = rotate(img.cpu().numpy()[0,0], angle, reshape=False)
                rotated_img_tensor = torch.tensor(rotated_img).unsqueeze(0).unsqueeze(0).to(device)
                
                logits = model(rotated_img_tensor)
                prob_map_rot = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
                # Restore the rotation to the original
                prob_map_rot = rotate(prob_map_rot, -angle, reshape=False)
                prob_maps.append(prob_map_rot)

            # Average probability of rotations
            prob_map = np.mean(prob_maps, axis=0)

            # Calculating metrics for 10 thresholds
            for t in thresholds:
                pred = (prob_map > t).astype(np.uint8)
                gt   = (mask_np > 0).astype(np.uint8)
                valid = mask_np != 255

                metrics[t]["pixel"].append(pixel_error(pred[valid], gt[valid]))
                metrics[t]["rand"].append(rand_error(pred[valid], gt[valid]))
                metrics[t]["warp"].append(warping_error_normalized(pred, gt))

                # Precision and Recall at threshold t
                TP = np.logical_and(pred[valid]==1, gt[valid]==1).sum()
                FP = np.logical_and(pred[valid]==1, gt[valid]==0).sum()
                FN = np.logical_and(pred[valid]==0, gt[valid]==1).sum()

                metrics[t]["prec"].append(TP/(TP+FP+1e-6))
                metrics[t]["rec"].append(TP/(TP+FN+1e-6))

                # Display some sample predictions
                if k < num_samples_to_show:
                    plot_image(img, mask_np, pred)
                    k+=1

    # Aggregate over all thresholds
    best_t = min(metrics, key=lambda t: np.mean(metrics[t]["rand"]))
    best_scores = {k: np.mean(v) for k, v in metrics[best_t].items()}

    # Average results
    mean_warp = best_scores["warp"]
    mean_rand = best_scores["rand"]
    mean_pix  = best_scores["pixel"]
    mean_prec = best_scores["prec"]
    mean_rec  = best_scores["rec"]

    print("\n======= EM TEST RESULTS =======")
    print(f"Best Threshold (Rand): {best_t:.3f}")
    print(f"Warping Error: {mean_warp:.6f}")
    print(f"Rand Error:    {mean_rand:.6f}")
    print(f"Pixel Error:   {mean_pix:.6f}")
    print(f"Precision:     {mean_prec:.4f}")
    print(f"Recall:        {mean_rec:.4f}")
    print("===============================")

    return mean_warp, mean_rand, mean_pix, mean_prec, mean_rec


def test_cell_tracking_unet(save_path, test_ds, device, threshold=0.25, num_samples_to_show=2):
    # Unpack hyperparameters
    model, batch_size = load_checkpoint(save_path, device)
    model.eval()

    # Loaders
    num_workers = 0
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    iou_scores, dice_scores = [], []
    precisions_scores, recalls_scores = [], []

    with torch.no_grad():
        for i, (img, mask, _) in enumerate(test_loader):
            img  = img.to(device)
            mask = mask.to(device)

            # Forward
            logits = model(img)

            # Cas 2 canaux → foreground prob
            probs = torch.softmax(logits, dim=1)[:, 1]

            prob_map = probs.cpu().numpy()[0]
            mask_np  = mask.cpu().numpy()[0]

            # Binarize frontiere
            binary = prob_map > threshold

            # Distance transform
            distance = distance_transform_edt(binary)

            # Find local maxima
            local_max = peak_local_max(distance, labels=binary, footprint=np.ones((3, 3)),exclude_border=False)

            markers = np.zeros_like(distance, dtype=int)
            for idx, (r, c) in enumerate(local_max, start=1):
                markers[r, c] = idx

            # Apply watershed
            labels_ws = watershed(-distance, markers, mask=binary)

            # Conversion for metrics
            pred_classes = (labels_ws > 0).astype(np.uint8)

            # Compute metrics
            iou_scores.append(iou_score(pred_classes, mask_np))
            dice_scores.append(dice(pred_classes, mask_np))

            TP = np.logical_and(pred_classes==1, mask_np==1).sum()
            FP = np.logical_and(pred_classes==1, mask_np==0).sum()
            FN = np.logical_and(pred_classes==0, mask_np==1).sum()
            precisions_scores.append(TP/(TP+FP+1e-6))
            recalls_scores.append(TP/(TP+FN+1e-6))

            # Display some sample predictions
            if i < num_samples_to_show:
                plot_image(img, mask_np, pred_classes)

    # Average
    mean_iou   = np.mean(iou_scores)
    mean_dice  = np.mean(dice_scores)
    mean_prec  = np.mean(precisions_scores)
    mean_rec   = np.mean(recalls_scores)

    print("\n======= CELL TRACKING TEST RESULTS =======")
    print(f"IoU Score:   {mean_iou:.4f}")
    print(f"Dice Score:  {mean_dice:.4f}")
    print(f"Precision:   {mean_prec:.4f}")
    print(f"Recall:      {mean_rec:.4f}")
    print("===========================================")

    return mean_iou, mean_dice, mean_prec, mean_rec