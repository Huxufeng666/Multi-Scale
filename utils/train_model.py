
from tqdm import tqdm
import torch
import os
import csv
import torchvision.utils as vutils
import torch.nn.functional as F



def train_base_loss(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    # 使用 tqdm 添加进度条
    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs ,encoder_loss= model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)




def train_EncoerCosSimilarity_loss(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    # 使用 tqdm 添加进度条
    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs, ncoer_loss_cos = model(images)
        loss_seg = criterion(outputs, masks)
        
        if ncoer_loss_cos is not None:
            loss = loss_seg + 0.2 * ncoer_loss_cos   # λ=0.2 可调
        else:
            loss = loss_seg
                    
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)




def train_EncoerCosSimilarity_loss(model, train_loader, criterion, optimizer, device, lambda_cos=0.2):
    model.train()
    epoch_loss = 0.0
    
    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs, ncoer_loss_cos = model(images)
        loss_seg = criterion(outputs, masks)

        if ncoer_loss_cos is not None:
            if torch.is_tensor(ncoer_loss_cos):
                ncoer_loss_cos = ncoer_loss_cos.mean()
            loss = loss_seg + lambda_cos * ncoer_loss_cos
        else:
            loss = loss_seg

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)



class Trainer_Base:
    def __init__(self, model, train_loader, criterion, optimizer, device, lambda_cos=0.2):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lambda_cos = lambda_cos

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs , loss_se= self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
 
            total_loss += loss.item()

        return total_loss / len(self.train_loader)


class Trainer_EncoerCosSimilarity:
    def __init__(self, model, train_loader, criterion, optimizer, device, lambda_cos=0.2):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lambda_cos = lambda_cos

    def compute_loss_se(self, features):
        # 检查 None 或空 tensor
        if features is None or features.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 自动扩展维度，如果是一维 tensor，变成 [batch_size, 1]
        if features.dim() == 1:
            features = features.unsqueeze(0)  # 变成 [1, feature_dim]

        # 如果是标量，也直接返回 0
        if features.dim() != 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # L2 normalize
        features = F.normalize(features, p=2, dim=1)

        # 计算 cosine similarity 矩阵
        sim_matrix = torch.matmul(features, features.t())

        # mask 去掉对角线
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)

        loss_se = 1 - sim_matrix.mean()
        return loss_se

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # 模型输出: outputs 和 features（特征向量用于计算 loss_se）
            outputs, features = self.model(inputs)

            # 主任务 loss
            loss_main = self.criterion(outputs, targets)

            # self-embedding loss
            loss_se = self.compute_loss_se(features)

            # 总 loss
            loss = loss_main + self.lambda_cos * loss_se

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)
    
    

class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device,
                 aux_weights=None, warmup_epochs=5, max_epochs=100):
        """
        aux_weights: dict of individual loss weights, e.g.
            {
               "loss_sim": 0.1,
               "loss_b": 0.5,
               "loss_f": 0.5,
               "ef_loss": 0.25  # can be used to multiply sum of ef losses
            }
        warmup_epochs: how many epochs only use main loss
        """
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion  # main segmentation loss, e.g. BCEWithLogitsLoss/Dice+BCE
        self.optimizer = optimizer
        self.device = device

        # 默认权重（可由外部传入）
        default = {"loss_sim": 0.1, "loss_b": 0.5, "loss_f": 0.5, "ef": 0.25}
        if aux_weights is None:
            aux_weights = default
        else:
            # 把没给的键设默认
            for k, v in default.items():
                aux_weights.setdefault(k, v)
        self.aux_weights = aux_weights

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch  # 0-based to 1-based

    def _aux_scale(self):
        """动态 scale：从 0 线性增长到 1（基于 warmup_epochs）"""
        if self.warmup_epochs <= 0:
            return 1.0
        return min(self.current_epoch / float(self.warmup_epochs), 1.0)

    def train_one_epoch(self):
        
        
          # ===== 🔹 动态调整 loss_b / loss_f 权重 =====
        if self.current_epoch < self.warmup_epochs:
            # 前 warmup_epochs 个 epoch：关闭背景监督，只学前景
            self.aux_weights["loss_b"] = 0.0
            self.aux_weights["loss_f"] = 1.0
        else:
            # warmup 之后逐步恢复背景权重，从 0 → 0.5 线性增长
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            self.aux_weights["loss_b"] = 0.5 * min(progress, 1.0)
            self.aux_weights["loss_f"] = 1.0

        # 方便你在 tqdm 中看到动态调整情况
        print(f"[Epoch {self.current_epoch}] loss_b={self.aux_weights['loss_b']:.3f}, loss_f={self.aux_weights['loss_f']:.3f}")
        
        
        self.model.train()
        total_loss = 0.0
        total_main = 0.0
        total_aux = 0.0
        aux_scale = self._aux_scale()
        epochs_completed = self.current_epoch + 1
        pbar = tqdm(self.train_loader, desc=f"Epoch {epochs_completed} training", leave=False)
        
        
        
        for batch in pbar:
            images, gts = batch
            images = images.to(self.device)
            gts = gts.to(self.device)

            self.optimizer.zero_grad()

            # 假设 model 返回 (final_output, aux_losses_dict)
            # aux_losses_dict 里可能包含 keys: 'loss_sim', 'loss_b', 'loss_f', 'ef_losses'（list 或 dict）
            outputs, aux_losses = self.model(images, gt=gts)

            # main loss (expect outputs are logits; criterion handles logits or sigmoid accordingly)
            main_loss = self.criterion(outputs, gts)

            # 计算 aux loss 合计：兼容模型直接给 total_aux 或分开给多个 loss 的情况
            # aux_losses 可能是:
            # - 一个单一 tensor (total_aux_loss)
            # - 或一个 dict 包含各项 losses
            aux_loss_total = torch.tensor(0.0, device=self.device)

            if aux_losses is not None:
                if isinstance(aux_losses, torch.Tensor):
                    aux_loss_total = aux_losses
                elif isinstance(aux_losses, dict):
                    # accumulate using configured weights
                    # 支持 ef 层返回多个 loss 的情形
                    # 安全取用：若某项不存在，就跳过
                    if "loss_sim" in aux_losses:
                        aux_loss_total = aux_loss_total + self.aux_weights["loss_sim"] * aux_losses["loss_sim"]
                    # decouple losses（可能只有 loss_b, loss_f）
                    if "loss_b" in aux_losses:
                        aux_loss_total = aux_loss_total + self.aux_weights["loss_b"] * aux_losses["loss_b"]
                    if "loss_f" in aux_losses:
                        aux_loss_total = aux_loss_total + self.aux_weights["loss_f"] * aux_losses["loss_f"]
                    # ef 层可能以列表/tuple形式返回每层 loss，或者一个单一值
                    if "ef_losses" in aux_losses:
                        ef = aux_losses["ef_losses"]
                        if isinstance(ef, (list, tuple)):
                            ef_sum = sum([e for e in ef if e is not None])
                        else:
                            ef_sum = ef
                        aux_loss_total = aux_loss_total + self.aux_weights.get("ef", 0.0) * ef_sum
                    # 如果模型内部已经给了 total_aux_loss 的话也兼容
                    if "total_aux" in aux_losses:
                        aux_loss_total = aux_loss_total + aux_losses["total_aux"]

            # 使用 aux_scale 逐渐将辅助 loss 加入到总 loss 中
            loss = main_loss * 0.2 + aux_scale  *  aux_loss_total * 0.3

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_main += main_loss.item()
            total_aux += aux_loss_total.item() if isinstance(aux_loss_total, torch.Tensor) else float(aux_loss_total)

            pbar.set_postfix({
                "loss": f"{total_loss/(len(pbar)+1):.4f}",
                "main": f"{total_main/(len(pbar)+1):.4f}",
                "aux": f"{total_aux/(len(pbar)+1):.4f}",
                "aux_scale": f"{aux_scale:.2f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_main = total_main / len(self.train_loader)
        avg_aux = total_aux / len(self.train_loader)
        return {"avg_loss": avg_loss, "avg_main": avg_main, "avg_aux": avg_aux}
