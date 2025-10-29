
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class BUS_UCLM_DATASET(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))

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
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
