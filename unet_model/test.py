import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet_model.model import UNet
from skimage import segmentation
from scipy.ndimage import binary_dilation
from sklearn.metrics import adjusted_rand_score


# -------------------------------
#   METRICS
# -------------------------------

def warping_error(pred, target):
    """
    Simple warping error based on contour mismatches
    pred, target: H x W, integers
    """
    # Extract edges
    pred_edges = segmentation.find_boundaries(pred, mode='outer')
    target_edges = segmentation.find_boundaries(target, mode='outer')
    
    # Dilate edges to tolerate small misalignments
    pred_edges_dil = binary_dilation(pred_edges)
    target_edges_dil = binary_dilation(target_edges)
    
    # Count mismatched edge pixels
    mismatch = np.logical_xor(pred_edges_dil, target_edges_dil)
    return mismatch.sum() / target_edges_dil.sum()  # proportion d’erreur

def rand_error(pred, target):
    """
    Computes Rand error between two label masks
    Returns 1 - Adjusted Rand Index (so error = 0 if perfect)
    """
    return 1 - adjusted_rand_score(target.flatten(), pred.flatten())

def pixel_error(pred, target):
    """
    Compute the pixel-wise error.
    pred, target: H x W (or B x H x W), integers (class labels)
    Returns error rate between 0 and 1
    """
    return np.mean(pred != target)

def iou_score(pred, target, eps=1e-6):
    """Compute the IoU (Intersection over Union) between prediction and target mask."""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection  # Compute union
    return (intersection + eps) / (union + eps)

def dice(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    return (2*inter + eps) / (pred.sum() + target.sum() + eps)


# -------------------------------
#   TEST FUNCTION
# -------------------------------

def test_model(
    model_dir,
    model_save,
    test_ds,
    device,
    batch_size=1,
    num_samples_to_show=1
):
    """
    Evaluate a trained UNet model on a test dataset.

    Args:
        model_path: path to the trained .pth file
        test_ds: PyTorch test dataset
        device: "cuda" or "cpu"
        num_samples_to_show: number of sample predictions to display
    """
    # Load test dataset
    #num_workers = max(1, min(8, os.cpu_count() // 2))
    num_workers = 0
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Create path
    save_path = os.path.join(model_dir, model_save)
    
    # Load model and move it to the device
    model = UNet().to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()                        # Set to evaluation mode (disables dropout, batchnorm, etc.)

    print("\nModel loaded. Starting evaluation...\n")

    # Lists to store metrics
    warp_scores = []
    rand_scores = []
    pix_scores = []
    iou_scores = []
    dice_scores = []
    precisions_scores = []
    recalls_scores = []

    # -------------------------------
    #   Evaluation Loop
    # -------------------------------

    with torch.no_grad():                               # Disable gradient computation for evaluation
        for i, (img, mask) in enumerate(test_loader):
            img = img.to(device)                        # Move input to device
            mask = mask.to(device).long()               # Convert mask to long type for classification

            # Forward pass
            pred = model(img)                           # Model prediction
            pred_classes = torch.argmax(pred, dim=1).cpu().numpy()[0]  # Predicted class per pixel
            mask_np = mask.cpu().numpy()[0]             # Ground truth mask as numpy array

            # Compute metrics
            valid_mask = mask_np != 255
            warp_scores.append(warping_error(pred_classes[valid_mask], mask_np[valid_mask]))
            rand_scores.append(rand_error(pred_classes[valid_mask], mask_np[valid_mask]))
            pix_scores.append(pixel_error(pred_classes[valid_mask], mask_np[valid_mask]))
            iou_scores.append(iou_score(pred_classes[valid_mask], mask_np[valid_mask]))
            dice_scores.append(dice(pred_classes[valid_mask], mask_np[valid_mask]))
            # Precision / Recall
            TP = np.logical_and(pred_classes[valid_mask] == 1, mask_np[valid_mask] == 1).sum()
            FP = np.logical_and(pred_classes[valid_mask] == 1, mask_np[valid_mask] == 0).sum()
            FN = np.logical_and(pred_classes[valid_mask] == 0, mask_np[valid_mask] == 1).sum()
            precision = TP / (TP + FP + 1e-6)
            recall    = TP / (TP + FN + 1e-6)
            precisions_scores.append(precision)
            recalls_scores.append(recall)

            # Display some sample predictions
            if i < num_samples_to_show:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img.cpu().numpy()[0][0], cmap="gray")  # Original image
                axes[0].set_title("Image")
                axes[0].axis("off")

                axes[1].imshow(mask_np, cmap="gray")    # Ground truth mask
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred_classes, cmap='gray')             # Predicted mask
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()

    # -------------------------------
    #   Final Metrics
    # -------------------------------
    
    mean_warp = np.mean(warp_scores)    # Average warping error
    mean_rand = np.mean(rand_scores)    # Average Rand error
    mean_pix = np.mean(pix_scores)      # Average pixel error
    mean_iou = np.mean(iou_scores)      # Average IoU
    mean_dice = np.mean(dice_scores)    # Average dice
    mean_precision = np.mean(precisions_scores)    # Average dice
    mean_recall = np.mean(recalls_scores)    # Average dice

    # Print results
    print("\n======= TEST RESULTS =======")
    print(f"Warping Error:      {mean_warp:.4f}")
    print(f"Rand Error:         {mean_rand:.4f}")
    print(f"Pixel Error:        {mean_pix:.4f}")
    print(f"IoU Score:          {mean_iou:.4f}")
    print(f"Dice Score:         {mean_dice:.4f}")
    print(f"Precision Score:    {mean_precision:.4f}")
    print(f"Recall Score:       {mean_recall:.4f}")
    print("============================")

    return mean_warp, mean_rand, mean_pix, mean_iou, mean_dice, mean_precision, mean_recall