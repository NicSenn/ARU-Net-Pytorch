import os
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from random import uniform

class NewDataset(Dataset):
    def __init__(self, image_dir, mask_dir, random_downsampling, image_height, image_width):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.random_downsampling = random_downsampling
        self.image_height = image_height
        self.image_width = image_width
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # Bozen Dataset
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_GT0.png"))
        # CBAD Dataset
        mask_path = ''
        if(self.images[index].endswith('.jpg')):
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        elif(self.images[index].endswith('.JPG')): # some pictures are stored as ".JPG"
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".JPG", ".png"))

        # Training Images as RGB
        #image = np.array(Image.open(img_path).convert("RGB"))
        # Greyscale
        image = Image.open(img_path).convert("L")
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        image = transform(image)
        image=np.array(image, dtype=np.float32)

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # L = greyscale
        mask[mask == 255.0] = 1.0 # because we do a sigmoid on the last activation , we make sure that the correct labels are used

        if self.random_downsampling:

            if image.shape[0] > self.image_height*2 or image.shape[1] > self.image_width*2:
                resize = A.LongestMaxSize(max_size=self.image_height*2, p=1)
                downscaled_version = resize(image=image, mask=mask)
                image = downscaled_version["image"]
                mask = downscaled_version["mask"]

            # random downscaling
            aScale = uniform(0.2, 0.5)
            tempImage = Image.fromarray(image)
            size = tuple((np.array(tempImage.size) * aScale).round().astype(int))
            image = np.array(tempImage.resize(size, Image.NEAREST), dtype=np.float32)

            tempMask = Image.fromarray(mask)
            mask = np.array(tempMask.resize(size, Image.NEAREST), dtype=np.float32)

            # normalization
            norm = A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0)
            downscaled_version = norm(image=image, mask=mask)
            image = downscaled_version["image"]
            mask = downscaled_version["mask"]

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
                mask = np.pad(mask, npad, mode='constant', constant_values=0)

            # convert to tensor
            augmentation = A.Compose([ToTensorV2()])
            augmentations = augmentation(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            transform = A.Compose(
                [

                    A.Resize(height=self.image_height, width=self.image_width),
                    A.Normalize(mean = [0.0], std=[1.0], max_pixel_value=255.0), # getting a value between 0 and 1
                    # automatically converts correct image format from PIL.Image (NHWC) to PyTorch format (NCHW)
                    ToTensorV2(),
                ],
            )

            augmentations = transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
