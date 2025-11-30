import torch
import monai
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity,
    Resize, Compose, Lambda
)
import matplotlib.pyplot as plt

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== LOAD MODEL ==========
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model_path = r"C:\Users\Aymen\.spyder-py3\unet_busi_cleaned.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model Loaded Successfully!")

# ========== TEST IMAGE ==========
test_image_path = r"C:\MesEtudes\SAR\MLA\cleaned_data\malignant\malignant (10).png"

# ========== TRANSFORMS ==========
test_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Lambda(lambda x: x[0:1]),  # RGB → grayscale
    ScaleIntensity(),
    Resize((256, 256)),
])

# ========== LOAD IMAGE ==========
image = test_transforms(test_image_path)
image_tensor = image.clone().detach().unsqueeze(0).to(device)

# ========== PREDICT ==========
with torch.no_grad():
    output = model(image_tensor)
    pred = torch.argmax(output, dim=1).cpu()[0]

# ========== DISPLAY RESULTS ==========
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image[0], cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(pred, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(image[0], cmap="gray")
plt.imshow(pred, alpha=0.4, cmap="jet")

plt.tight_layout()
plt.show()
