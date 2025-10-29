import os
import time
import argparse
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# === Import your modules ===
from model import UNet_EncoderLoss, UNetTransformer, FullModel,DFSANet
from data import BUSI_DADASET, BUSBRA_DATASET, BUS_UCLM_DATASET
from utils.TrainingLogger import TrainingLogger ,append_csv_averages
from utils.save_sample import save_sample_segmentations
from utils.validate_model import validate_model
from utils import Trainer, Validator


# ============================================================
# ✅ 1. 固定随机种子，确保完全可复现
# ============================================================
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA 11+ 需要

def worker_init_fn(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

# ============================================================
# ✅ 2. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train U-Net for ultrasound image segmentation')
    parser.add_argument('--data_path', type=str, default='/workspace/dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda or cpu')
    parser.add_argument('--output_dir', type=str, default='/workspace/result', help='Directory to save results')
    args = parser.parse_args()

    # 固定种子（必须在任何 DataLoader、模型创建之前）
    setup_seed(42)

    # === Dataset Paths ===
    train_image_dir = os.path.join(args.data_path, 'train', 'images')
    train_mask_dir = os.path.join(args.data_path, 'train', 'masks')
    val_image_dir = os.path.join(args.data_path, 'val', 'images')
    val_mask_dir = os.path.join(args.data_path, 'val', 'masks')

    # === Datasets & Loaders ===
    train_dataset = BUSI_DADASET(train_image_dir, train_mask_dir)
    val_dataset = BUSI_DADASET(val_image_dir, val_mask_dir)

    # ⚠️ 保证每次 DataLoader 输出顺序一致（shuffle=True 也会被同样的种子控制）
    g = torch.Generator()
    g.manual_seed(42)  # PyTorch DataLoader 内部 shuffle 用

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,              # 开启打乱，但由种子控制顺序固定
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=worker_init_fn
    )

    # === Model, Loss, Optimizer ===
    device = torch.device(args.device)
    model = DFSANet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # === Output Structure ===
    dataset_name = os.path.basename(args.data_path)
    model_name = model.__class__.__name__
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    specific_output_dir = os.path.join(args.output_dir, f"{dataset_name}-{model_name}", timestamp)
    weights_dir = os.path.join(specific_output_dir, 'weights')
    log_dir = os.path.join(specific_output_dir, 'logs')

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # === Logger ===
    logger = TrainingLogger(log_dir=log_dir)
    best_val_loss = float('inf')

    # === Trainer & Validator ===
    trainer = Trainer(model, train_loader, criterion, optimizer, device)
    validator = Validator(model, val_loader, criterion, device)

    print("\n--- Start Training ---\n")

    # ===== 训练主循环 =====
    top_k = 3
    patience = 10
    no_improve_epochs = 0
    saved_models = []
        
    for epoch in range(args.epochs):
        start_time = time.time()

        # ======== 训练阶段 ========
        train_loss = trainer.train_one_epoch()

        # ======== 验证阶段 ========
        val_loss, overall, tumor, normal, tumor_stat = validator.validate()

        time_taken = time.time() - start_time

        # ======== 日志记录 ========

        logger.log(epoch + 1,train_loss, val_loss,time_taken,val_metrics=overall["all"],val_metrics_tumor=tumor["tumor"],val_metrics_normal=normal["normal"], tumor_in_gt=tumor_stat)
        
        # ======== 保存当前模型（按 epoch 命名） ========
        model_file = os.path.join(weights_dir, f"model_epoch{epoch+1}_val{val_loss:.4f}.pth")
        torch.save(model.state_dict(), model_file)

        # ======== 仅保留 top-3 最优模型 ========
        saved_models.append((val_loss, model_file))
        saved_models.sort(key=lambda x: x[0])  # 按 loss 升序
        if len(saved_models) > top_k:
            _, to_delete = saved_models.pop(-1)
            if os.path.exists(to_delete):
                os.remove(to_delete)
                print(f"🗑️ Deleted old model: {to_delete}")

        # ======== 检查是否改进 ========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0

            best_model_path = os.path.join(weights_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New best model saved to {best_model_path}")

            # 绘制曲线并保存样例
            logger.plot_losses()
            logger.plot_metrics()
            # logger.plot_metrics1()
            
            sample_save_path = os.path.join(specific_output_dir, f'samples/epoch_{epoch + 1}')
            save_sample_segmentations(
                model, val_dataset, sample_save_path,
                device, epoch + 1, sample_indices=[0, 1, 2, 3]
            )
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs} epoch(s).")
        
        # ======== 提前停止机制 ========
        if no_improve_epochs >= patience:
            print(f"⛔ Early stopping triggered after {patience} epochs without improvement.")
            break
    
    append_csv_averages(logger.csv_path)        


# ============================================================
# 🚀 Entry Point
# ============================================================
if __name__ == '__main__':
    main()












