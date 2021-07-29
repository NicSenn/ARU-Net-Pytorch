import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TestDataset(Dataset):
    def __init__(self, image_dir, image_height, image_width, padding):
        self.image_dir = image_dir
        self.image_height = image_height
        self.image_width = image_width
        self.padding = padding
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # RGB
        #image = np.array(Image.open(img_path).convert("RGB"))
        # Greyscale
        image = Image.open(img_path).convert("L")
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        image = transform(image)
        image=np.array(image, dtype=np.float32)

        if self.padding: 

            # downscale
            max_size = max(self.image_height, self.image_width)
            resize = A.LongestMaxSize(max_size=max_size, p=1)
            downscaled_version = resize(image=image)
            image = downscaled_version["image"]

            # normalization
            norm = A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0)
            downscaled_version = norm(image=image)
            image = downscaled_version["image"]

            # padding
            maxH = self.image_height
            maxW = self.image_width
            padH = maxH - image.shape[0]
            padW = maxW - image.shape[1]
            if padH + padW > 0:
                if padH < 0:
                    padH = 0
                if padW < 0:
                    padW = 0
                npad = ((0, padH), (0, padW))
                image = np.pad(image, npad, mode='constant', constant_values=1)

            augmentation = A.Compose([ToTensorV2()])
            augmentations = augmentation(image=image)
            image = augmentations["image"]

        else:

            test_transform = A.Compose([
                A.Resize(height=self.image_height, width=self.image_width),
                #A.Normalize(mean = [0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0), # getting a value between 0 and 1
                A.Normalize(mean = [0.0], std=[1.0], max_pixel_value=255.0), # getting a value between 0 and 1
                # automatically converts correct image format from PIL.Image (NHWC) to PyTorch format (NCHW)
                ToTensorV2()])

            augmentations = test_transform(image = image)
            image = augmentations["image"]
        
        return image
