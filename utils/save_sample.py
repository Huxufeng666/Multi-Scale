
import os
import torch
import torchvision.transforms as T
from PIL import Image


def denormalize(tensor, mean, std):
    """反标准化张量"""
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    tensor = tensor.clone() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def concat_and_save(image, mask, pred_mask, save_path):
    """
    拼接原图、真实mask和预测mask并保存
    image: tensor (C,H,W)
    mask: tensor (1,H,W)
    pred_mask: tensor (1,H,W)
    """
    to_pil = T.ToPILImage()

    # ---- 1. 反标准化原图 ----
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    image_denormalized = denormalize(image.cpu(), mean=IMG_MEAN, std=IMG_STD)
    img_pil = to_pil(image_denormalized)
    img_pil_gray = img_pil.convert('L')  # 转为灰度图

    # ---- 2. mask 和 pred_mask 转灰度图 ----
    mask_pil = to_pil(mask.cpu())
    pred_pil = to_pil(pred_mask.cpu())

    # ---- 3. 拼接图像 ----
    w, h = img_pil_gray.size
    concat_img = Image.new('L', (w * 3, h))  # 灰度拼接画布

    concat_img.paste(img_pil_gray, (0, 0))
    concat_img.paste(mask_pil, (w, 0))
    concat_img.paste(pred_pil, (w * 2, 0))

    concat_img.save(save_path)


def save_sample_segmentations(model, val_dataset, save_dir, device, epoch, sample_indices=[0, 1, 2, 3]):
    """
    保存原图 + GT + 预测mask的拼接结果
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    # concat_dir = os.path.join(save_dir, f'epoch_{epoch}')
    # os.makedirs(concat_dir, exist_ok=True)

    to_pil = T.ToPILImage()

    with torch.no_grad():
        for idx in sample_indices:
            image, mask = val_dataset[idx]
            img_name = val_dataset.images[idx] if hasattr(val_dataset, 'images') else f'sample_{idx}.png'

            image_tensor = image.unsqueeze(0).to(device)
            output,loss = model(image_tensor)

            # 将预测结果二值化并去掉batch维度
            pred = (output > 0.5).float().cpu().squeeze(0)

            # 保存三图拼接结果
            save_path = os.path.join(save_dir, f'{os.path.basename(img_name)}')
            concat_and_save(image, mask, pred, save_path)

 