


from model import UNet_EncoderLoss ,UNetTransformer, FullModel,DFSANet
from data import BUSI_DADASET, BUSBRA_DATASET, BUS_UCLM_DATASET

import os
import argparse
import glob
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm



# ======== 评估函数 ========
def evaluate(model, dataloader, device, metrics_csv=None, image_names=None, concat_dir=None, threshold=0.5):
    """评估模型，保存拼接图和指标 CSV，包括汇总统计"""
    # 内部工具函数
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean, device=tensor.device).view(-1,1,1)
        std = torch.tensor(std, device=tensor.device).view(-1,1,1)
        tensor = tensor.clone() * std + mean
        return torch.clamp(tensor, 0, 1)

    def concat_and_save(image, mask, pred_mask, save_path):
        to_pil = T.ToPILImage()
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        img_denorm = denormalize(image.cpu(), IMG_MEAN, IMG_STD)
        img_pil = to_pil(img_denorm).convert('L')
        mask_pil = to_pil(mask.cpu())
        pred_pil = to_pil(pred_mask.cpu())

        w, h = img_pil.size
        concat_img = Image.new('L', (w*3, h))
        concat_img.paste(img_pil, (0,0))
        concat_img.paste(mask_pil, (w,0))
        concat_img.paste(pred_pil, (w*2,0))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        concat_img.save(save_path)

    model.eval()
    results = []
    eps = 1e-7
    if image_names is None:
        image_names = [f"img_{i}" for i in range(len(dataloader.dataset))]

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", ncols=100)):
            # 兼容 tuple / list 或 dict
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                masks = batch[1].to(device)
                if len(batch) > 2:
                    filenames = batch[2]
                else:
                    start_idx = idx * dataloader.batch_size
                    filenames = image_names[start_idx:start_idx + images.size(0)]
            else:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                filenames = batch.get('filename', image_names[idx*dataloader.batch_size:(idx+1)*dataloader.batch_size])

            pred  ,loss= model(images)
            preds = torch.sigmoid(pred)
            
            if preds.shape[-2:] != masks.shape[-2:]:
                preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            preds = (preds - preds.min()) / (preds.max() - preds.min() + eps)
            masks_norm = (masks - masks.min()) / (masks.max() - masks.min() + eps)
            preds_bin = (preds > threshold).float()
            masks_bin = (masks > 0.5).float()

            for img, pred, mask, fname in zip(images, preds_bin, masks_bin, filenames):
                pred_bool = pred.squeeze().cpu().bool()
                mask_bool = mask.squeeze().cpu().bool()

                # 保存拼接图
                if concat_dir:
                    save_path = os.path.join(concat_dir, fname)
                    concat_and_save(img, mask.squeeze(), pred.squeeze(), save_path)

                # Case 判断
                if mask_bool.sum() == 0 and pred_bool.sum() == 0:
                    results.append({'Image':fname,'Dice':1.0,'Precision':1.0,'Recall':1.0,'IoU':1.0,'Valid':False})
                    continue
                if mask_bool.sum() > 0 and pred_bool.sum() == 0:
                    results.append({'Image':fname,'Dice':0.0,'Precision':0.0,'Recall':0.0,'IoU':0.0,'Valid':True})
                    continue
                if mask_bool.sum() == 0 and pred_bool.sum() > 0:
                    results.append({'Image':fname,'Dice':0.0,'Precision':0.0,'Recall':0.0,'IoU':0.0,'Valid':True})
                    continue

                intersection = (pred_bool & mask_bool).sum().item()
                union = (pred_bool | mask_bool).sum().item()
                dice = (2*intersection)/(pred_bool.sum().item()+mask_bool.sum().item()+eps)
                precision = intersection/(pred_bool.sum().item()+eps)
                recall = intersection/(mask_bool.sum().item()+eps)
                iou = intersection/(union+eps)

                results.append({'Image':fname,'Dice':dice,'Precision':precision,'Recall':recall,'IoU':iou,'Valid':True})

    # 汇总统计
    valid_results = [r for r in results if r['Valid']]
    empty_correct = len([r for r in results if not r['Valid']])
    total = len(results)
    mean_dice = np.mean([r['Dice'] for r in valid_results]) if valid_results else 0.0
    mean_precision = np.mean([r['Precision'] for r in valid_results]) if valid_results else 0.0
    mean_recall = np.mean([r['Recall'] for r in valid_results]) if valid_results else 0.0
    mean_iou = np.mean([r['IoU'] for r in valid_results]) if valid_results else 0.0

    print(f"📊 Evaluation Summary")
    print(f" - 有效样本数 (含肿瘤): {len(valid_results)} / {total}")
    print(f" - 空样本正确率: {empty_correct / total:.2%}")
    print(f" - Dice: {mean_dice:.4f}")
    print(f" - Precision: {mean_precision:.4f}")
    print(f" - Recall: {mean_recall:.4f}")
    print(f" - IoU: {mean_iou:.4f}")


        
    summary_csv = os.path.join(concat_dir if concat_dir else '.', '*summary_metrics.csv')
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Metric', 'Value'])
        writer.writeheader()
        writer.writerow({'Metric': 'Valid Samples', 'Value': f"{len(valid_results)} / {total}"})
        writer.writerow({'Metric': 'Empty Sample Accuracy', 'Value': f"{empty_correct:.2%}"})
        writer.writerow({'Metric': 'Mean Dice', 'Value': f"{mean_dice:.4f}"})
        writer.writerow({'Metric': 'Mean Precision', 'Value': f"{mean_precision:.4f}"})
        writer.writerow({'Metric': 'Mean Recall', 'Value': f"{mean_recall:.4f}"})
        writer.writerow({'Metric': 'Mean IoU', 'Value': f"{mean_iou:.4f}"})    
            
        

    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Image','Dice','Precision','Recall','IoU','Valid'])
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            'Image':'Summary',
            'Dice':mean_dice,
            'Precision':mean_precision,
            'Recall':mean_recall,
            'IoU':mean_iou,
            'Valid':f'EmptyAcc={empty_correct/total:.2%}'
        })

    return {'Dice':mean_dice, 'Precision':mean_precision, 'Recall':mean_recall, 'IoU':mean_iou}

# ======== Main 函数 ========
def main():
    parser = argparse.ArgumentParser(description='Test U-Net model')
    parser.add_argument('--data_path', type=str, default='dataset', help='Path to dataset')
    parser.add_argument('--weights', type=str, default='result/dataset-DFSANet/20251024_042655/weights/model_epoch98_val0.0530.pth', help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    args = parser.parse_args()

    val_image_dir = os.path.join(args.data_path, 'test', 'images')
    val_mask_dir = os.path.join(args.data_path, 'test', 'masks')
    val_dataset = BUS_UCLM_DATASET(val_image_dir, val_mask_dir)  # 自定义 Dataset
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = DFSANet().to(device)  # 自定义模型
    model.load_state_dict(torch.load(args.weights, map_location=device))

    weights_directory = os.path.dirname(args.weights)
    concat_dir = os.path.join(weights_directory, 'results_concat')
    metrics_csv = os.path.join(concat_dir, '*metrics.csv')
    val_image_names = sorted(os.listdir(val_image_dir))

    metrics = evaluate(model, val_loader, device, metrics_csv=metrics_csv, image_names=val_image_names, concat_dir=concat_dir)
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()