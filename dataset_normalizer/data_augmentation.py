import numpy as np
import cv2
from PIL import Image

def elastic_deformation_3x3(image, mask, sigma=10, p=0.7):
    """
    Elastic deformation as described in the original UNet paper.
    
    - Random displacement vectors on a coarse 3x3 grid
    - Displacements sampled from N(0, sigma)
    - Bicubic interpolation to full resolution
    
    Args:
        image (H,W) numpy array
        mask  (H,W) numpy array
        sigma (float): std deviation of displacement in pixels
        p (float): probability to apply deformation
    
    Returns:
        deformed image, deformed mask
    """
    # --- 1. Apply function with probability ---
    if np.random.rand() > p:
        return image, mask

    H, W = image.shape

    # --- 2. Generate 3x3 displacement grid ---
    dx_small = np.random.normal(0, sigma, (3, 3)).astype(np.float32)
    dy_small = np.random.normal(0, sigma, (3, 3)).astype(np.float32)

    # --- 3. Upscale to full resolution using bicubic interpolation ---
    dx = cv2.resize(dx_small, (W, H), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy_small, (W, H), interpolation=cv2.INTER_CUBIC)

    # --- 4. Create coordinate grid ---
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # --- 5. Apply deformation ---
    img_deformed = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask_deformed = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return img_deformed, mask_deformed


def random_rotate_shift(image, mask, max_angle=30, max_shift=5, p=0.8):

    # --- 1. Apply function with probability ---
    if np.random.rand() > p:
        return image, mask
    
    # --- 2. Sample random angle and shifts ---
    angle = np.random.normal(0, max_angle)
    tx = np.random.normal(0, max_shift)
    ty = np.random.normal(0, max_shift)

    from scipy.ndimage import rotate, shift

    # --- 3. Apply rotation and shift ---
    image = rotate(image, angle, reshape=False, order=3, mode="reflect")
    mask  = rotate(mask, angle, reshape=False, order=0)

    image = shift(image, shift=(ty, tx), order=3, mode="reflect")
    mask  = shift(mask, shift=(ty, tx), order=0)

    return image, mask


def intensity_variation(image):
    # Convert PIL image to NumPy array (float32 in range [0,1])
    image_np = np.array(image).astype(np.float32) / 255.0

    # Apply intensity variation
    gain = np.random.uniform(0.9, 1.1)
    bias = np.random.uniform(-0.05, 0.05)
    image_np = image_np * gain + bias

    # Clip values to [0,1] to avoid overflow
    image_np = np.clip(image_np, 0.0, 1.0)
    # Convert back to PIL Image (uint8)
    image_out = Image.fromarray((image_np * 255).astype(np.uint8))

    return image_out


def compute_dropout_rate(dataset_size):
    """
    Exponentially decaying dropout:
    - f(0)   = 0.5
    - f(∞)   = 0
    - form : f(x) = 0.5 * exp(-x/tau)
    - f(500) = 0.25
    - Result : tau = 500 / log(2)
    """
    tau = 500 / np.log(2)         
    return 0.5 * np.exp(-dataset_size / tau)