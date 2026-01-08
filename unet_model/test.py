import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, rotate, distance_transform_edt
from sklearn.metrics import adjusted_rand_score
from skimage.segmentation import watershed, find_boundaries
from skimage.feature import peak_local_max


# -------------------------------
#   METRICS
# -------------------------------

def warping_error(pred, target):
    """
    Simple warping error based on contour mismatches
    pred, target: H x W, integers
    """
    # Extract edges
    pred_edges = find_boundaries(pred, mode='outer')
    target_edges = find_boundaries(target, mode='outer')
    
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
#   TESTS FUNCTIONS
# -------------------------------
def test_em_unet(model, dataloader, device, thresholds=np.linspace(0.0, 1.0, 10), rot_angles = [0, 45, 90, 135, 180, 225, 270], num_samples_to_show=2):
    model.eval()

    pixel_scores, rand_scores, warp_scores = [], [], []
    precisions_scores, recalls_scores = [], []

    with torch.no_grad():
        for i, (img, mask, _) in enumerate(dataloader):
            img = img.to(device)
            mask_np = mask.cpu().numpy()[0]  # ground truth pour cette image

            # 7 rotations
            prob_maps = []
            for angle in rot_angles:
                rotated_img = rotate(img.cpu().numpy()[0,0], angle, reshape=False)
                rotated_img_tensor = torch.tensor(rotated_img).unsqueeze(0).unsqueeze(0).to(device)
                
                logits = model(rotated_img_tensor)
                prob_map_rot = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
                # remettre la rotation à l’original
                prob_map_rot = rotate(prob_map_rot, -angle, reshape=False)
                prob_maps.append(prob_map_rot)

            # moyenne des probabilités sur les rotations
            prob_map = np.mean(prob_maps, axis=0)

            # calcul des métriques pour 10 seuils
            for t in thresholds:
                pred = (prob_map > t).astype(np.uint8)
                valid_mask = mask_np != 255

                # Pixel-wise metrics
                warp_scores.append(warping_error(pred[valid_mask], mask_np[valid_mask]))
                rand_scores.append(rand_error(pred[valid_mask], mask_np[valid_mask]))
                pixel_scores.append(pixel_error(pred[valid_mask], mask_np[valid_mask]))

                # Optional precision / recall
                TP = np.logical_and(pred[valid_mask]==1, mask_np[valid_mask]==1).sum()
                FP = np.logical_and(pred[valid_mask]==1, mask_np[valid_mask]==0).sum()
                FN = np.logical_and(pred[valid_mask]==0, mask_np[valid_mask]==1).sum()
                precisions_scores.append(TP/(TP+FP+1e-6))
                recalls_scores.append(TP/(TP+FN+1e-6))

            # Display some sample predictions
            if i < num_samples_to_show:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(img.cpu().numpy()[0, 0], cmap="gray")
                axes[0].set_title("Image")
                axes[0].axis("off")

                axes[1].imshow(mask_np, cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred, cmap="gray")
                axes[2].set_title("Prediction (Watershed)")
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()

    # Average results
    mean_warp  = np.mean(warp_scores)
    mean_rand  = np.mean(rand_scores)
    mean_pix   = np.mean(pixel_scores)
    mean_prec  = np.mean(precisions_scores)
    mean_rec   = np.mean(recalls_scores)

    print("\n======= EM TEST RESULTS =======")
    print(f"Warping Error: {mean_warp:.6f}")
    print(f"Rand Error:    {mean_rand:.6f}")
    print(f"Pixel Error:   {mean_pix:.6f}")
    print(f"Precision:     {mean_prec:.4f}")
    print(f"Recall:        {mean_rec:.4f}")
    print("===============================")

    return mean_warp, mean_rand, mean_pix, mean_prec, mean_rec


def test_cell_tracking_unet(model, dataloader, device, threshold=0.5, num_samples_to_show=2):
    model.eval()

    iou_scores, dice_scores = [], []
    precisions_scores, recalls_scores = [], []

    with torch.no_grad():
        for i, (img, mask, _) in enumerate(dataloader):
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
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(img.cpu().numpy()[0, 0], cmap="gray")
                axes[0].set_title("Image")
                axes[0].axis("off")

                axes[1].imshow(mask_np, cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred_classes, cmap="gray")
                axes[2].set_title("Prediction (Watershed)")
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()

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