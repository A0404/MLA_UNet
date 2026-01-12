import numpy as np
from skimage.measure import label
from skimage.segmentation import find_boundaries
from sklearn.metrics import adjusted_rand_score
from scipy.ndimage import distance_transform_edt

# ---------- PIXEL ERROR -----------------
def pixel_error(pred, gt, ignore_value=255):
    """
    Pixel-wise classification error.
    """
    valid = gt != ignore_value
    return np.mean(pred[valid] != gt[valid])

# ---------- RAND ERROR -----------------
def induced_segmentation(binary_membrane):
    """
    Given a binary membrane map (1 = membrane),
    returns connected regions of non-membrane space.
    """
    background = binary_membrane == 0
    return label(background, connectivity=1)

def rand_error(pred_membrane, gt_membrane):
    """
    Rand error computed on the induced segmentation.
    """
    seg_pred = induced_segmentation(pred_membrane)
    seg_gt   = induced_segmentation(gt_membrane)

    return 1.0 - adjusted_rand_score(
        seg_gt.flatten(),
        seg_pred.flatten()
    )

# ---------- WARPING ERROR -----------------
def warping_error(pred, gt):
    """
    ISBI-like Warping Error (Boundary Displacement Error)

    pred, gt : binary masks (0 = membrane, 1 = cell)
    Returns a float (lower is better)
    """

    # 1. Extract boundaries (membranes)
    gt_bound = find_boundaries(gt, mode='inner')
    pred_bound = find_boundaries(pred, mode='inner')

    # 2. If no boundaries detected, avoid division by zero
    if gt_bound.sum() == 0 or pred_bound.sum() == 0:
        return np.nan

    # 3. Distance transform of predicted boundaries
    dist_pred = distance_transform_edt(~pred_bound)

    # 4. Measure distance from each GT boundary pixel to nearest predicted boundary
    distances = dist_pred[gt_bound]

    # 5. Average distance (normalized by image diagonal if desired)
    return distances.mean()

def warping_error_normalized(pred, gt):
    h, w = gt.shape
    diag = np.sqrt(h*h + w*w)
    return warping_error(pred, gt) / diag

# ---------- DICE COEFFICIENT -----------------
def dice(pred, gt, ignore_value=255):
    """
    Dice coefficient for binary masks.
    """
    valid = gt != ignore_value
    pred = pred[valid]
    gt   = gt[valid]

    intersection = np.sum(pred * gt)
    size_sum = np.sum(pred) + np.sum(gt)

    if size_sum == 0:
        return 1.0

    return (2.0 * intersection) / size_sum

# ---------- IOU SCORE -----------------
def iou_score(pred, gt, ignore_value=255):
    """
    Intersection over Union (IoU) score for binary masks.
    """
    valid = gt != ignore_value
    pred = pred[valid]
    gt   = gt[valid]

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection

    if union == 0:
        return 1.0

    return intersection / union