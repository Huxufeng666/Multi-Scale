

import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class BUSBRA_DATASET(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_files = sorted(os.listdir(image_dir))  # 图像文件名列表

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('bus', 'mask'))

        image = Image.open(img_path).convert('L')   # 灰度图像
        mask = Image.open(mask_path).convert('L')    # 二值图像

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask