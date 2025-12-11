import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

def compute_image_stats(img):
    """
    img: numpy array grayscale, valeurs en [0,1] ou [0,255]
    Retour: dict avec mean, std, rms_contrast, min, max, p2, p98, skew
    """
    mean = img.mean()
    std = img.std()
    rms = np.sqrt(np.mean((img - img.mean())**2))   # RMS contrast = std (same)
    p2, p98 = np.percentile(img, (2, 98))
    mn, mx = img.min(), img.max()
    skew = float(stats.skew(img.flatten()))
    return {"mean": float(mean), "std": float(std), "rms": float(rms),
            "min": float(mn), "max": float(mx), "p2": float(p2), "p98": float(p98), "skew": skew}

def plot_image_and_hist(img, title="image"):
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].axis("off")
    axes[0].set_title(title)
    axes[1].hist(img.flatten(), bins=256, range=(0,1))
    axes[1].set_title("Histogramme")
    plt.show()

def contrast_stretch(img, low_perc=2, high_perc=98):
    """Contrast stretching by percentiles. img in [0,1] or [0,255]."""
    lo, hi = np.percentile(img, [low_perc, high_perc])
    if hi - lo < 1e-6:
        return img
    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out

def gamma_correction(img, gamma=1.0):
    """Gamma correction. gamma<1 brighten, >1 darken."""
    out = np.power(img, gamma)
    out = np.clip(out, 0.0, 1.0)
    return out

def clahe_cv2(img, clipLimit=2.0, tileGridSize=(8,8)):
    """CLAHE via OpenCV. Input either [0,1] or [0,255]. Returns [0,1]."""
    a = img.astype(np.float32)
    scaled = a if a.max() > 1.0 else (a * 255.0)
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    out_u8 = clahe.apply(scaled)
    return out_u8.astype(np.float32) / 255.0

def compare_contrast_methods(img, methods=None, titles=None, show_hist=True):
    """
    img: numpy grayscale image in [0,1] or [0,255]
    methods: list of callables f(img)->img
    titles: list of titles
    """
    a = img
    if methods is None:
        methods = [
            lambda x: x,  # identity
            lambda x: contrast_stretch(x, 2, 98),
            lambda x: clahe_cv2(x, clipLimit=2.0, tileGridSize=(8,8)),
            lambda x: gamma_correction(x, gamma=0.8),  # brighten slightly
            lambda x: gamma_correction(x, gamma=1.2),  # darken slightly
        ]
        titles = ["orig", "stretch 2-98", "CLAHE (2.0)", "gamma 0.8", "gamma 1.2"]
    imgs = [m(img) for m in methods]
    n = len(imgs)
    plt.figure(figsize=(4*n,4))
    for i, im in enumerate(imgs):
        a = im if im.max() > 1.0 else (im)
        plt.subplot(2, n, i+1)
        plt.imshow(a, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(titles[i])
        plt.subplot(2, n, n + i + 1)
        plt.hist(a.flatten(), bins=256, range=(0,1))
        plt.title("hist")
    plt.show()

    # print stats
    for t, im in zip(titles, imgs):
        print(t, compute_image_stats(im))