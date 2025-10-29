from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, jaccard_score
import torchvision.transforms as T



def validate_base_loss(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        # 使用 tqdm 添加进度条
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs,encoder_loss = model(images)
            loss_seg = criterion(outputs, masks)
            
            if encoder_loss is not None:
                loss = loss_seg + 0.2 * encoder_loss   # λ=0.2 可调
            else:
                loss = loss_seg
            
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def validate_model(model, val_loader, criterion, device, use_aux_loss=True):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # forward
            outputs = model(images)
            encoder_loss = None
            if isinstance(outputs, tuple):  # 模型可能返回 (seg, aux_loss)
                outputs, encoder_loss = outputs
            
            # 计算主分割损失
            loss_seg = criterion(outputs, masks)
            
            # 可选附加 loss
            if use_aux_loss and encoder_loss is not None:
                loss = loss_seg + 0.2 * encoder_loss
            else:
                loss = loss_seg
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    return total_loss / total_samples





# ============================================================
# 🧩 Validator Class
# ============================================================


# class Validator:
#     def __init__(self, model, val_loader, criterion, device):
#         self.model = model
#         self.val_loader = val_loader
#         self.criterion = criterion
#         self.device = device

#     def validate(self):
#         self.model.eval()
#         total_loss = 0.0

#         # 分别记录有肿瘤和无肿瘤样本的指标
#         dice_all, precision_all, recall_all, iou_all = [], [], [], []
#         dice_tumor, precision_tumor, recall_tumor, iou_tumor = [], [], [], []
#         dice_normal, precision_normal, recall_normal, iou_normal = [], [], [], []

#         tumor_in_gt = []  # 记录每张图片真实是否有肿瘤

#         with torch.no_grad():
#             for batch in tqdm(self.val_loader, desc="Validating", leave=False):
#                 inputs, targets = batch
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs, _ = self.model(inputs)

#                 loss = self.criterion(outputs, targets)
#                 total_loss += loss.item()

#                 preds = torch.sigmoid(outputs)
#                 preds_bin = (preds > 0.5).float()

#                 # --- metric calculation ---
#                 eps = 1e-7
#                 intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
#                 union = (preds_bin + targets).sum(dim=(1, 2, 3)) - intersection

#                 dice = (2 * intersection + eps) / (preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
#                 precision = (intersection + eps) / (preds_bin.sum(dim=(1, 2, 3)) + eps)
#                 recall = (intersection + eps) / (targets.sum(dim=(1, 2, 3)) + eps)
#                 iou = (intersection + eps) / (union + eps)

#                 dice_all.extend(dice.cpu().numpy())
#                 precision_all.extend(precision.cpu().numpy())
#                 recall_all.extend(recall.cpu().numpy())
#                 iou_all.extend(iou.cpu().numpy())

#                 # --- 根据目标是否有肿瘤，分开统计 ---
#                 for idx, t in enumerate(targets):
#                     if t.sum() > 0:  # 有肿瘤
#                         tumor_in_gt.append("Yes")
#                         dice_tumor.append(dice[idx].item())
#                         precision_tumor.append(precision[idx].item())
#                         recall_tumor.append(recall[idx].item())
#                         iou_tumor.append(iou[idx].item())
#                     else:  # 无肿瘤
#                         tumor_in_gt.append("No")
#                         dice_normal.append(dice[idx].item())
#                         precision_normal.append(precision[idx].item())
#                         recall_normal.append(recall[idx].item())
#                         iou_normal.append(iou[idx].item())

#         avg_loss = total_loss / len(self.val_loader)

#         # --- 计算平均指标 ---
#         metrics = {
#             "all": {
#                 "dice": np.mean(dice_all),
#                 "precision": np.mean(precision_all),
#                 "recall": np.mean(recall_all),
#                 "iou": np.mean(iou_all)
#             },
#             "tumor": {
#                 "dice": np.mean(dice_tumor) if dice_tumor else None,
#                 "precision": np.mean(precision_tumor) if precision_tumor else None,
#                 "recall": np.mean(recall_tumor) if recall_tumor else None,
#                 "iou": np.mean(iou_tumor) if iou_tumor else None
#             },
#             "normal": {
#                 "dice": np.mean(dice_normal) if dice_normal else None,
#                 "precision": np.mean(precision_normal) if precision_normal else None,
#                 "recall": np.mean(recall_normal) if recall_normal else None,
#                 "iou": np.mean(iou_normal) if iou_normal else None
#             }
#         }

#         return avg_loss, metrics, tumor_in_gt


class Validator:
    def __init__(self, model, val_loader, criterion, device):
        """
        Args:
            model: 已训练的模型
            val_loader: 验证数据 DataLoader
            criterion: 损失函数
            device: 设备 ('cuda' or 'cpu')
        """
        self.model = model
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        # --- 各种指标收集 ---
        dice_scores, precisions, recalls, ious = [], [], [], []
        dice_tumor, prec_tumor, rec_tumor, iou_tumor = [], [], [], []
        dice_normal, prec_normal, rec_normal, iou_normal = [], [], [], []
        tumor_in_gt = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                preds = torch.sigmoid(outputs)
                preds_bin = (preds > 0.5).float()

                eps = 1e-7
                intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
                union = (preds_bin + targets).sum(dim=(1, 2, 3)) - intersection
                dice = (2 * intersection + eps) / (preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
                precision = (intersection + eps) / (preds_bin.sum(dim=(1, 2, 3)) + eps)
                recall = (intersection + eps) / (targets.sum(dim=(1, 2, 3)) + eps)
                iou = (intersection + eps) / (union + eps)

                # 区分肿瘤样本 / 正常样本
                for i in range(len(targets)):
                    if targets[i].sum() > 0:  # 有肿瘤
                        dice_tumor.append(dice[i].item())
                        prec_tumor.append(precision[i].item())
                        rec_tumor.append(recall[i].item())
                        iou_tumor.append(iou[i].item())
                        tumor_in_gt.append("Yes")
                    else:  # 无肿瘤
                        dice_normal.append(dice[i].item())
                        prec_normal.append(precision[i].item())
                        rec_normal.append(recall[i].item())
                        iou_normal.append(iou[i].item())
                        tumor_in_gt.append("No")

                # 全部样本整体指标
                dice_scores.extend(dice.cpu().numpy())
                precisions.extend(precision.cpu().numpy())
                recalls.extend(recall.cpu().numpy())
                ious.extend(iou.cpu().numpy())

        # --- 计算平均值 ---
        avg_metrics = {
            "dice": np.mean(dice_scores),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "iou": np.mean(ious)
        }

        metrics_tumor = {
            "dice": np.mean(dice_tumor) if dice_tumor else 0.0,
            "precision": np.mean(prec_tumor) if prec_tumor else 0.0,
            "recall": np.mean(rec_tumor) if rec_tumor else 0.0,
            "iou": np.mean(iou_tumor) if iou_tumor else 0.0
        }

        metrics_normal = {
            "dice": np.mean(dice_normal) if dice_normal else 0.0,
            "precision": np.mean(prec_normal) if prec_normal else 0.0,
            "recall": np.mean(rec_normal) if rec_normal else 0.0,
            "iou": np.mean(iou_normal) if iou_normal else 0.0
        }

        avg_loss = total_loss / len(self.val_loader)

        # return avg_loss, avg_metrics, metrics_tumor, metrics_normal, tumor_in_gt
        return  avg_loss, {"all": avg_metrics}, {"tumor": metrics_tumor}, {"normal": metrics_normal}, tumor_in_gt