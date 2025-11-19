import torch
from torch.nn import Conv2d, ConvTranspose2d, ReLU, MaxPool2d
from torch import cat as Concatenate


### Copilot code ###
def Center_crop(tensor, target):
    """ Concatenate along the channel dimension. In PyTorch image tensors are (N, C, H, W).
        Here the encoder feature map may be larger by some pixels.
        We crop it centrally to match the upsampled tensor spatial size before concatenation to avoid shape mismatch.

        Center-crop `tensor` to `target_size` (th, tw).
        tensor: (N, C, H, W)
        target_size: (th, tw) """
    
    if tensor.shape[2:] != target.shape[2:]:
        _, _, h, w = tensor.shape
        th, tw = target.shape[2:]
        if h == th and w == tw:
            return tensor
        top = (h - th) // 2
        left = (w - tw) // 2
        tensor = tensor[:, :, top:top+th, left:left+tw]
    return tensor
### ------------- ###


def UNet_Model(input):
    # First Encoder Block
    conv572_570 = ReLU(Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)(input))
    conv570_568 = ReLU(Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)(conv572_570))
    pool568_284 = MaxPool2d(kernel_size=2, stride=2)(conv570_568)

    # Second Encoder Block
    conv284_282 = ReLU(Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)(pool568_284))
    conv282_280 = ReLU(Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)(conv284_282))
    pool280_140 = MaxPool2d(kernel_size=2, stride=2)(conv282_280)

    # Third Encoder Block
    conv140_138 = ReLU(Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)(pool280_140))
    conv138_136 = ReLU(Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)(conv140_138))
    pool136_68 = MaxPool2d(kernel_size=2, stride=2)(conv138_136)

    # Fourth Encoder Block
    conv68_66 = ReLU(Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)(pool136_68))
    conv66_64 = ReLU(Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)(conv68_66))
    pool64_32 = MaxPool2d(kernel_size=2, stride=2)(conv66_64)

    # Bottleneck
    conv32_30 = ReLU(Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)(pool64_32))
    conv30_28 = ReLU(Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=0)(conv32_30))

    # First Decoder Block
    up28_56 = ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)(conv30_28)   # Depth : 512
    conv66_64_cropped = Center_crop(conv66_64, up28_56)
    concat56_512 = Concatenate((conv66_64_cropped, up28_56), dim=1)     # Depth : 1024
    conv56_54 = ReLU(Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0)(concat56_512))  # Depth : 512
    conv54_52 = ReLU(Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)(conv56_54))

    # Second Decoder Block
    up52_104 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)(conv54_52)   # Depth : 256
    conv138_136_cropped = Center_crop(conv138_136, up52_104)
    concat104_256 = Concatenate((conv138_136_cropped, up52_104), dim=1)     # Depth : 512
    conv104_102 = ReLU(Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)(concat104_256))  # Depth : 256
    conv102_100 = ReLU(Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)(conv104_102))      

    # Third Decoder Block
    up100_200 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)(conv102_100)   # Depth : 128
    conv282_280_cropped = Center_crop(conv282_280, up100_200)
    concat200_128 = Concatenate((conv282_280_cropped, up100_200), dim=1)     # Depth : 256
    conv200_198 = ReLU(Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)(concat200_128))  # Depth : 128
    conv198_196 = ReLU(Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)(conv200_198))  

    # Fourth Decoder Block
    up196_392 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)(conv198_196)   # Depth : 64
    conv570_568_cropped = Center_crop(conv570_568, up196_392)
    concat392_64 = Concatenate((conv570_568_cropped, up196_392), dim=1)     # Depth : 128
    conv392_390 = ReLU(Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)(concat392_64))  # Depth : 64
    conv390_388 = ReLU(Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)(conv392_390))

    # Output Layer
    output = Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)(conv390_388)    # Depth : 2

    return output