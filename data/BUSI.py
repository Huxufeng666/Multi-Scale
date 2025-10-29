from PIL import Image
import os 
from torch.utils.data import Dataset

from torchvision import transforms
import cv2

import numpy as np  


class BUSI_DADASET(Dataset):
    def __init__(self, image_dir: str, mask_dir: str,
                 image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))

        # 图像变换
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # mask 变换（保持单通道）
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())  # 二值化
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("L")

        # --- 处理 mask ---
        base_name, _ = os.path.splitext(image_name)
        mask1_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        mask2_path = os.path.join(self.mask_dir, f"{base_name}_mask_1.png")

        mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask1_path) else None
        mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask2_path) else None

        if (mask1 is None) and (mask2 is None):
            raise FileNotFoundError(f"No mask found for {base_name}")

        # --- 合并为单通道 ---
        if mask1 is not None and mask2 is not None:
            mask = np.maximum(mask1, mask2)  # 并集
        elif mask1 is not None:
            mask = mask1
        else:
            mask = mask2

        mask = Image.fromarray(mask)

        # --- transform ---
        image = self.image_transform(image)   # (1, 256, 256)
        mask = self.mask_transform(mask)      # (1, 256, 256)

        return image, mask


