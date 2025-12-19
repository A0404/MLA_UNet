import matplotlib.pyplot as plt
import os

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