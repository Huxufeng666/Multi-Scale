
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupleLayer_base(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.fg = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.bg = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.bn_f = nn.BatchNorm2d(in_ch)
        self.bn_b = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=False)
        
        # (新增) 1. 添加损失函数
        # 我们使用 BCEWithLogitsLoss，因为它在数值上很稳定
        # 假设前景(F_map)应该匹配掩码(1)，背景(B_map)应该匹配反转的掩码(0)
        self.loss_fn = nn.BCEWithLogitsLoss()

    # (修改) 2. forward 方法现在接受 'mask'
    def forward(self, x, mask=None):
        
        # --- 3. 计算 Logits (在 ReLU 之前) ---
        # 我们需要在激活函数(ReLU)之前计算损失
        fg_logits = self.bn_f(self.fg(x))
        bg_logits = self.bn_b(self.bg(x))

        # --- (新增) 4. 计算损失 ---
        loss_f = torch.tensor(0.0, device=x.device)
        loss_b = torch.tensor(0.0, device=x.device)

        # 只有在 'train' 模式且 'mask' (gt) 被提供时，才计算损失
        if self.training and mask is not None:
            # 确保 mask 和特征图有相同的空间尺寸 (H, W)
            if mask.shape[2:] != x.shape[2:]:
                # (如果尺寸不同，使用 'nearest' 插值下采样 mask)
                gt_mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
            else:
                gt_mask = mask # (B, 1, H, W)
            
            # 计算前景损失: fg_logits 应该匹配 gt_mask
            # .expand_as() 将 (B, 1, H, W) 扩展为 (B, C, H, W)
            loss_f = self.loss_fn(fg_logits, gt_mask.expand_as(fg_logits))
            
            # 计算背景损失: bg_logits 应该匹配反转的 (1 - gt_mask)
            loss_b = self.loss_fn(bg_logits, (1 - gt_mask).expand_as(bg_logits))

        # --- 5. 应用 ReLU 以便输出激活后的特征 ---
        # (这是传递给下一层 SpatialAttention 的特征)
        F_map = self.relu(fg_logits)
        B_map = self.relu(bg_logits)
        
        # (修改) 6. 返回 4 个值
        return F_map, B_map, loss_b, loss_f











class DecoupleLayer(nn.Module):
    def __init__(self, in_ch, use_depthwise=True, pos_weight=None, use_soft_target=False):
        super().__init__()
        # 更稳健的 conv: depthwise + pointwise
        if use_depthwise:
            self.dw_fg = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
            self.pw_fg = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
            self.dw_bg = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
            self.pw_bg = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        else:
            self.dw_fg = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
            self.pw_fg = nn.Identity()
            self.dw_bg = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
            self.pw_bg = nn.Identity()

        self.bn_f = nn.BatchNorm2d(in_ch)
        self.bn_b = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

        # 推荐使用 BCEWithLogitsLoss，pos_weight 用于正样本较少的情况
        if pos_weight is not None:
            pos_w = torch.tensor(pos_weight)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.use_soft_target = use_soft_target

    def forward(self, x, mask=None):
        # conv
        fg = self.dw_fg(x)
        fg = self.pw_fg(fg)
        fg_logits = self.bn_f(fg)

        bg = self.dw_bg(x)
        bg = self.pw_bg(bg)
        bg_logits = self.bn_b(bg)

        loss_f = torch.tensor(0.0, device=x.device)
        loss_b = torch.tensor(0.0, device=x.device)

        if self.training and mask is not None:
            # 强制二值化并下采样到特征图大小
            gt_mask = F.interpolate(mask, size=fg_logits.shape[2:], mode='nearest')
            gt_mask = (gt_mask > 0.5).float()

            # 如果想用 soft target，可把 gt_mask 变成 0.9/0.1 而不是 1/0
            if self.use_soft_target:
                pos = 0.9
                neg = 0.1
                gt_pos = gt_mask * pos + (1 - gt_mask) * neg
                gt_neg = (1 - gt_mask) * pos + gt_mask * neg
            else:
                gt_pos = gt_mask
                gt_neg = 1 - gt_mask

            # expand target 到通道维度
            target_f = gt_pos.expand_as(fg_logits)
            target_b = gt_neg.expand_as(bg_logits)

            loss_f = self.loss_fn(fg_logits, target_f)
            loss_b = self.loss_fn(bg_logits, target_b)

        # **不要在传给下游的特征上用 ReLU**（会丢信息）
        # 如果后续需要概率显示，可以在可视化时用 sigmoid
        F_map = fg_logits  # or torch.sigmoid(fg_logits) IF downstream expects probabilities
        B_map = bg_logits

        return F_map, B_map, loss_b, loss_f
