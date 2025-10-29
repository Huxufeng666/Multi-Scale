import torch
import torch.nn as nn
import torch.nn.functional as F
from .CCA import CristCoistAttention, CCA_FusionModule
from utils.cos_loss import CosineSimLoss
from .Decouple import DecoupleLayer
from .Efficient_Features import EfficientFeature
import torch.nn.functional as F


# --- (为了让此代码可运行，我先粘贴您提供的组件占位符) ---
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_res = nn.Sequential(nn.MaxPool2d(2), BasicBlock(in_channels, out_channels))
    def forward(self, x): return self.pool_res(x)




class SimpleUp(nn.Module):
    """
    一个简单的 2 倍上采样模块，使用转置卷积 (ConvTranspose2d)。
    它可以选择接收一个 skip_feature 进行拼接和融合。
    """
    def __init__(self, in_channels, out_channels, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        if use_skip:
            fusion_in_channels = out_channels * 2
        else:
            fusion_in_channels = out_channels
            
        # 这里使用一个简单的 Conv -> BN -> ReLU 块来替代之前的 BasicBlock
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_below, x_skip=None):

        # 1. 上采样
        x_up = self.up(x_below)
        
        # 2. 拼接 (如果提供了 skip feature)
        if self.use_skip and x_skip is not None:
            
            # 尺寸安全检查 (与你之前的代码保持一致)
            if x_up.size() != x_skip.size():
                x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x_up, x_skip], dim=1)
        else:
            x = x_up

        # 3. 特征融合
        return self.conv_fusion(x)



class Up(nn.Module):
    def __init__(self, in_channels_below, skip_channels, out_channels):
        super().__init__()
        # 正确设置上采样的输入通道数为 in_channels_below (512)
        self.up = nn.ConvTranspose2d(in_channels_below, out_channels, kernel_size=2, stride=2)
        
        # res_block 的输入通道数是 (out_channels + skip_channels)
        self.res_block = BasicBlock(out_channels + skip_channels, out_channels) 
    def forward(self, x_below, x_skip):
        x_up = self.up(x_below)
        if x_up.size() != x_skip.size():
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_up, x_skip], dim=1)
        return self.res_block(x)
    
    


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)




class DecoupleLayer(nn.Module): # (您提供的真实版本)
    def __init__(self, in_ch):
        super().__init__()
        self.fg = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.bg = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.bn_f = nn.BatchNorm2d(in_ch)
        self.bn_b = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, x, mask=None):
        fg_logits = self.bn_f(self.fg(x))
        bg_logits = self.bn_b(self.bg(x))
        loss_f = torch.tensor(0.0, device=x.device)
        loss_b = torch.tensor(0.0, device=x.device)
        if self.training and mask is not None:
            gt_mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
            loss_f = self.loss_fn(fg_logits, gt_mask.expand_as(fg_logits))
            loss_b = self.loss_fn(bg_logits, (1 - gt_mask).expand_as(bg_logits))
        F_map = self.relu(fg_logits)
        B_map = self.relu(bg_logits)
        return F_map, B_map, loss_b, loss_f

class CosineSimLoss(nn.Module): # (占位符)
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)
    def forward(self, x, y):
        return (1 - self.cos(x, y)).mean()

# -----------------------------------------------------------
# 假设 2: 这是您上一节的 EfficientFeature 内部组件
# -----------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // 16, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(in_channels // 16, in_channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module): # (您上一节修改后的版本)
    def __init__(self, in_channels):
        super().__init__()
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.conv_sa = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, B, F):
        O_cat = torch.cat([B, F], dim=1)
        O_fused = self.fuse_conv(O_cat)
        avg_out = torch.mean(O_fused, dim=1, keepdim=True)
        max_out, _ = torch.max(O_fused, dim=1, keepdim=True)
        M = self.sigmoid(self.conv_sa(torch.cat([avg_out, max_out], dim=1)))
        return O_fused * M

# -----------------------------------------------------------
# (!! 关键 !!) 3. 重新定义 EfficientFeature 来匹配架构图
# -----------------------------------------------------------
class EfficientFeature(nn.Module):
    
    
    """
    这个模块严格按照图 d42f00.png 中的定义：
    1. 接收 3 个来自不同尺度的输入
    2. 将它们融合 (Resize -> Concat -> 1x1 Conv)
    3. 将融合后的特征通过 CA -> Decouple -> SA 管道
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        # 1. 融合器 (Consolidator)
        total_in_ch = sum(in_channels_list)
        self.consolidator = nn.Sequential(
            nn.Conv2d(total_in_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 您已有的 EfficientFeature 核心管道
        self.channel_att = ChannelAttention(out_channels)
        self.decouple = DecoupleLayer(out_channels)
        self.spatial_att = SpatialAttention(out_channels) # (使用 C -> C 版本)

    def forward(self, inputs, mask=None):
        # inputs 是一个包含 3 个特征图的列表 [f1, f2, f3]
        # 目标尺度是第一个特征图 f1 的尺度
        target_size = inputs[0].shape[2:]
        
        # 1. 调整所有输入的大小并拼接
        # resized_inputs = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in inputs]
        resized_inputs = [torch.nn.functional.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in inputs]
        x_cat = torch.cat(resized_inputs, dim=1)
        
        # 2. 融合 (Consolidate)
        x_cons = self.consolidator(x_cat)
        
        # 3. 运行核心管道
        x_ca = self.channel_att(x_cons)
        B, F, loss_b, loss_f = self.decouple(x_ca, mask)
        out = self.spatial_att(B, F)
        
        return out, loss_b, loss_f




class FullModel(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_c=64):
        super().__init__()
        self.n_classes = n_classes
        
        # ----------------- 1. 编码器 Encoder 1-4 -----------------
        # (我们使用 inc + 3xDown 来匹配图中的 4 个尺度输出)
        self.enc1_pre = BasicBlock(n_channels, base_c) # H
        self.enc1 = Down(base_c, base_c)              # -> e1 (B, 64, H/2)
        
        self.enc2 = Down(base_c, base_c * 2)          # -> e2 (B, 128, H/4)
        self.enc3 = Down(base_c * 2, base_c * 4)      # -> e3 (B, 256, H/8)
        self.enc4 = Down(base_c * 4, base_c * 8)      # -> e4 (B, 512, H/16)


        # ----------------- 2. CCA 1-4 -----------------
        self.cca1 = CristCoistAttention(base_c)
        self.cca2 = CristCoistAttention(base_c * 2)
        self.cca3 = CristCoistAttention(base_c * 4)
        self.cca4 = CristCoistAttention(base_c * 8)
        
        # ----------------- 3. 相似度损失路径 (黄/绿/蓝) -----------------
        # 黄色 Concat + 绿色 Fusion Features (我们用一个 1x1 Conv 实现)
        self.fusion_features = nn.Sequential(
            nn.Conv2d(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1, bias=False), # (512+512) -> 512
            nn.BatchNorm2d(base_c * 8),
            nn.ReLU(inplace=True)
        )
        self.similarity_loss = CosineSimLoss()

        # ----------------- 4. 瓶颈 Decouple (深棕色) -----------------
        self.bottleneck_decouple = DecoupleLayer(base_c * 8) # 512

        # ----------------- 5. 高效特征路径 (Efficient Feature 1-4) -----------------
        # (!! 关键 !!) 这里的 in_channels_list 必须匹配图中的 3 个输入
        
        # self.bottleneck_encoder = DecoupleLayer(base_c * 8) # 512
        
        self.enc1_pre = BasicBlock(n_channels, base_c) 
        self.up_Simple4 = SimpleUp(base_c * 16, base_c * 4, use_skip=False)
        self.up_Simple3 = SimpleUp(base_c * 4, base_c * 2, use_skip=False)
        self.up_Simple2 = SimpleUp(base_c * 2, base_c  , use_skip=False)
        self.up_Simple1 = SimpleUp(base_c  , 1  , use_skip=False)
        
        
        
        # EF4 (H/16) inputs: c4(512), decouple(512), c3(256)
        self.ef4 = EfficientFeature([base_c * 8, base_c * 8, base_c * 8], base_c * 4) # out: 512
        
        # EF3 (H/8) inputs: c3(256), EF4_out(512), c2(128)
        self.ef3 = EfficientFeature([base_c * 4, base_c * 4, base_c * 4], base_c * 2) # out: 256
        
        # EF2 (H/4) inputs: c2(128), EF3_out(256), c1(64)
        self.ef2 = EfficientFeature([base_c * 2, base_c * 2, base_c* 2], base_c )     # out: 128
        
        # EF1 (H/2) inputs: c1(64), EF2_out(128), EF2_out(?) - 图中箭头含糊
        # (我们假设是 c1, ef2_out, e1)
        self.ef1 = EfficientFeature([base_c, base_c, base_c], 1)           # out: 64

        # ----------------- 6. 解码器 Decoder 1-4 -----------------
        # Dec 4 (H/16) -> (我们用一个 BasicBlock 作为瓶颈解码器)
        # self.dec4 = BasicBlock(base_c * 8, base_c * 8) # 512 -> 512
        self.reduce_conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.reduce_conv2 = nn.Conv2d(256,128, kernel_size=1, stride=1)
        self.reduce_conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.reduce_conv5 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        
        
        
        self.dec4 = Up(base_c * 16 ,base_c * 4  , base_c * 4)   # (512+512) -> 256
               
        # Dec 3 (H/8) -> Up(dec4_out + ef4_out)
        self.dec3 = Up(base_c * 4 ,base_c * 2  , base_c * 2)   # (512+512) -> 256
        
        # Dec 2 (H/4) -> Up(dec3_out + ef3_out)
        self.dec2 = Up(base_c * 2 ,base_c , base_c )   # (256+256) -> 128
        
        # Dec 1 (H/2) -> Up(dec2_out + ef2_out)
        self.dec1 = Up(base_c ,1  , base_c)      # (128+128) -> 64

        # ----------------- 7. 最终输出 -----------------
        self.final_out = OutConv(base_c, n_classes) # (来自 Dec 1)
        # EF1 的输出也需要上采样到 H
        self.final_ef1_out = OutConv(base_c, n_classes)
        

    def forward(self, x, gt=None):
        # 准备 gt (如果需要)
        gt_down16 = None
        if self.training and gt is not None:
            # gt_down16 = F.interpolate(gt, size=x.shape[2:]//16, mode='nearest')
            target_size_16 = (x.shape[2] // 16, x.shape[3] // 16)
            gt_down16 = F.interpolate(gt, size=target_size_16, mode='nearest')

        # --- 1. Encoder ---
        x_pre = self.enc1_pre(x) # H
        e1 = self.enc1(x_pre)    # H/2
        e2 = self.enc2(e1)       # H/4
        e3 = self.enc3(e2)       # H/8
        e4 = self.enc4(e3)       # H/16
        
        # --- 2. CCA ---
        c1 = self.cca1(e1)
        c2 = self.cca2(e2)
        c3 = self.cca3(e3)
        c4 = self.cca4(e4)
        
        # --- 3. Similarity Loss ---
        yellow_concat = torch.cat([e4, c4], dim=1)
        fusion_out = self.fusion_features(yellow_concat)
        
        loss_sim = torch.tensor(0.0, device=x.device)
        if self.training and gt is not None:
            
            gt_expanded = gt_down16.expand_as(e4)
            loss_sim_e4 = self.similarity_loss(e4, gt_expanded)
            loss_sim_c4 = self.similarity_loss(c4, gt_expanded)
            loss_sim = loss_sim_e4 + loss_sim_c4

        # --- 4. Bottleneck Decouple ---
    
        b_map, f_map, loss_b, loss_f = self.bottleneck_decouple(e4, gt_down16)
        d_in = b_map + f_map # (B, 512, H/16) - 这是 Decoder 4 的输入
        
        # --- 5. Efficient Feature Path (级联) ---
        # (我们必须为每个 EF 模块提供 mask)
 
        gt_h8 = None
        gt_h4 = None
        gt_h2 = None
        if self.training and gt is not None:
            gt_h8 = F.interpolate(gt, size=(x.shape[2] // 8, x.shape[3] // 8), mode='nearest')
            gt_h4 = F.interpolate(gt, size=(x.shape[2] // 4, x.shape[3] // 4), mode='nearest')
            gt_h2 = F.interpolate(gt, size=(x.shape[2] // 2, x.shape[3] // 2), mode='nearest')
        
        
        
        
        
        d4 = self.up_Simple4(yellow_concat)   # (B, 512, H/16)
        d3 = self.up_Simple3(d4)              # (B, 256, H/8)
        d2 = self.up_Simple2(d3)              # (B, 128, H/4)
        d1 = self.up_Simple1(d2)              # (B, 64,  H/2)

        # === 2. Efficient Feature Blocks (EF4 → EF1) ===
        # EF4 对应 H/16
        ef4_out, ef4_loss_b, ef4_loss_f = self.ef4([c4, d_in, fusion_out], gt_down16)
        ef4_up = F.interpolate(ef4_out, scale_factor=2, mode='bilinear', align_corners=False)  # → H/8

        # EF3 对应 H/8
        ef3_out, ef3_loss_b, ef3_loss_f = self.ef3([c3, ef4_up, d4], gt_h8)
        ef3_up = F.interpolate(ef3_out, scale_factor=2, mode='bilinear', align_corners=False)  # → H/4

        # EF2 对应 H/4
        ef2_out, ef2_loss_b, ef2_loss_f = self.ef2([c2, ef3_up, d3], gt_h4)
        ef2_up = F.interpolate(ef2_out, scale_factor=2, mode='bilinear', align_corners=False)  # → H/2

        # EF1 对应 H/2
        ef1_out, ef1_loss_b, ef1_loss_f = self.ef1([c1, ef2_up, d2], gt_h2)



        # === 3. Decoder 多源融合 ===

        #  Decoder4: 拼接 d4 + ef4_out → dec4
        dec4_out = self.dec4(yellow_concat, ef4_up)

        #  Decoder3: 拼接 dec4_out + ef3_out → dec3

        dec3_out = self.dec3(dec4_out, ef3_out)
       
        
        #  Decoder2: 拼接 dec3_out + ef2_out → dec2
    
        dec2_out = self.dec2(dec3_out, ef2_out)
       
         
        #  Decoder1: 拼接 dec2_out + ef1_out → dec1

        dec1_out = self.dec1(dec2_out, ef1_out)  # 若dec1不需skip可传None或略过

        # === 4. 输出层 ===

                            
        logits_d1 = self.final_out(dec1_out) # (B, 1, H/2)
        
        # logits_ef1 = self.final_ef1_out(ef1_out) # (B, 1, H/2)
        # (最终输出是 Dec 1 和 EF 1 的输出相加)
        # final_logits = logits_d1 + logits_ef1
        
        # 上采样到原始尺寸
        final_output = F.interpolate(logits_d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # --- 8. 收集所有损失 ---
        total_aux_loss = (loss_sim + loss_b + loss_f + 
                          ef4_loss_b + ef4_loss_f + 
                          ef3_loss_b + ef3_loss_f +
                          ef2_loss_b + ef2_loss_f +
                          ef1_loss_b + ef1_loss_f)
        
        return final_output, total_aux_loss
    
    
if __name__ == "__main__":
    pass
