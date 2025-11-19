import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
#  1. Fonction de Center Crop 
# --------------------------------------------------
def center_crop(feature_map, target_tensor):
    _, _, h, w = feature_map.shape
    _, _, th, tw = target_tensor.shape

    delta_h = h - th
    delta_w = w - tw

    top = delta_h // 2
    left = delta_w // 2

    return feature_map[:, :, top:top+th, left:left+tw]

# --------------------------------------------------
### ------------- ###
# --------------------------------------------------
#  2. Bloc de base : Double Convolution (sans padding)
# --------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------
#  3. ENCODER 
#     Chaque étape = 2 conv + maxpool (sauf bottleneck)
# --------------------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.enc1(x)
        x = self.pool(s1)

        s2 = self.enc2(x)
        x = self.pool(s2)

        s3 = self.enc3(x)
        x = self.pool(s3)

        s4 = self.enc4(x)
        x = self.pool(s4)

        x = self.bottleneck(x)

        return x, [s1, s2, s3, s4]
# --------------------------------------------------
#  4. DECODER 
#     upsampling + cropping + concatenation + conv
# --------------------------------------------------
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

    def forward(self, x, skips):
        s1, s2, s3, s4 = skips

        x = self.up1(x)
        s4 = center_crop(s4, x)
        x = torch.cat([s4, x], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        s3 = center_crop(s3, x)
        x = torch.cat([s3, x], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        s2 = center_crop(s2, x)
        x = torch.cat([s2, x], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        s1 = center_crop(s1, x)
        x = torch.cat([s1, x], dim=1)
        x = self.dec4(x)

        return x


# --------------------------------------------------
# --------------------------------------------------
#  5. MODÈLE COMPLET U-NET 
# --------------------------------------------------
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.final_conv(x)
        return x