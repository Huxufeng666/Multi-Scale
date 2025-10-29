import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # === 记录字段 ===
        self.epochs = []
        self.train_losses = []      # total loss
        self.train_main_losses = [] # main loss
        self.train_aux_losses = []  # aux loss
        self.val_losses = []

        # 整体指标
        self.val_dice = []
        self.val_precision = []
        self.val_recall = []
        self.val_iou = []

        # 肿瘤样本指标
        self.val_dice_tumor = []
        self.val_precision_tumor = []
        self.val_recall_tumor = []
        self.val_iou_tumor = []

        # 正常样本指标
        self.val_dice_normal = []
        self.val_precision_normal = []
        self.val_recall_normal = []
        self.val_iou_normal = []

        # === 创建 CSV 文件 ===
        self.csv_path = os.path.join(log_dir, f'training_log_{self.timestamp}.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch',
                'Train Loss', 'Main Loss', 'Aux Loss', 'Val Loss',
                'Dice', 'Precision', 'Recall', 'IoU',
                'Dice(Tumor)', 'Precision(Tumor)', 'Recall(Tumor)', 'IoU(Tumor)',
                'Dice(Normal)', 'Precision(Normal)', 'Recall(Normal)', 'IoU(Normal)',
                'Time(s)', 'Tumor Present'
            ])

    def log(self, epoch, train_stats, val_loss, time_taken,
            val_metrics=None, val_metrics_tumor=None, val_metrics_normal=None, tumor_in_gt=None):
        """
        train_stats: dict from Trainer.train_one_epoch()
                     keys: {'avg_loss', 'avg_main', 'avg_aux'}
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_stats.get('avg_loss', float('nan')))
        self.train_main_losses.append(train_stats.get('avg_main', float('nan')))
        self.train_aux_losses.append(train_stats.get('avg_aux', float('nan')))
        self.val_losses.append(val_loss)

        # 整体指标
        self.val_dice.append(val_metrics.get('dice') if val_metrics else float('nan'))
        self.val_precision.append(val_metrics.get('precision') if val_metrics else float('nan'))
        self.val_recall.append(val_metrics.get('recall') if val_metrics else float('nan'))
        self.val_iou.append(val_metrics.get('iou') if val_metrics else float('nan'))

        # 肿瘤样本指标
        self.val_dice_tumor.append(val_metrics_tumor.get('dice') if val_metrics_tumor else float('nan'))
        self.val_precision_tumor.append(val_metrics_tumor.get('precision') if val_metrics_tumor else float('nan'))
        self.val_recall_tumor.append(val_metrics_tumor.get('recall') if val_metrics_tumor else float('nan'))
        self.val_iou_tumor.append(val_metrics_tumor.get('iou') if val_metrics_tumor else float('nan'))

        # 正常样本指标
        self.val_dice_normal.append(val_metrics_normal.get('dice') if val_metrics_normal else float('nan'))
        self.val_precision_normal.append(val_metrics_normal.get('precision') if val_metrics_normal else float('nan'))
        self.val_recall_normal.append(val_metrics_normal.get('recall') if val_metrics_normal else float('nan'))
        self.val_iou_normal.append(val_metrics_normal.get('iou') if val_metrics_normal else float('nan'))

        # 统计验证集中肿瘤数量
        if tumor_in_gt is not None:
            num_tumor = tumor_in_gt.count("Yes")
            num_total = len(tumor_in_gt)
            tumor_stat = f"{num_tumor}/{num_total}"
        else:
            tumor_stat = ''

        # === 写入 CSV ===
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{self.train_losses[-1]:.6f}",
                f"{self.train_main_losses[-1]:.6f}",
                f"{self.train_aux_losses[-1]:.6f}",
                f"{val_loss:.6f}",
                f"{self.val_dice[-1]:.4f}",
                f"{self.val_precision[-1]:.4f}",
                f"{self.val_recall[-1]:.4f}",
                f"{self.val_iou[-1]:.4f}",
                f"{self.val_dice_tumor[-1]:.4f}",
                f"{self.val_precision_tumor[-1]:.4f}",
                f"{self.val_recall_tumor[-1]:.4f}",
                f"{self.val_iou_tumor[-1]:.4f}",
                f"{self.val_dice_normal[-1]:.4f}",
                f"{self.val_precision_normal[-1]:.4f}",
                f"{self.val_recall_normal[-1]:.4f}",
                f"{self.val_iou_normal[-1]:.4f}",
                f"{time_taken:.2f}",
                tumor_stat
            ])

    def log_average_metrics(self):
        """在 CSV 文件末尾记录整体、肿瘤样本、正常样本的平均指标"""
        def mean_or_nan(lst):
            return np.nanmean(lst) if lst else float('nan')

        overall = [mean_or_nan(self.val_dice), mean_or_nan(self.val_precision),
                   mean_or_nan(self.val_recall), mean_or_nan(self.val_iou)]
        tumor = [mean_or_nan(self.val_dice_tumor), mean_or_nan(self.val_precision_tumor),
                 mean_or_nan(self.val_recall_tumor), mean_or_nan(self.val_iou_tumor)]
        normal = [mean_or_nan(self.val_dice_normal), mean_or_nan(self.val_precision_normal),
                  mean_or_nan(self.val_recall_normal), mean_or_nan(self.val_iou_normal)]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(['# --- Overall Metrics ---', '', '', f"{overall[0]:.4f}", f"{overall[1]:.4f}",
                             f"{overall[2]:.4f}", f"{overall[3]:.4f}"])
            writer.writerow(['# --- Tumor Sample Metrics ---', '', '', f"{tumor[0]:.4f}", f"{tumor[1]:.4f}",
                             f"{tumor[2]:.4f}", f"{tumor[3]:.4f}"])
            writer.writerow(['# --- Normal Sample Metrics ---', '', '', f"{normal[0]:.4f}", f"{normal[1]:.4f}",
                             f"{normal[2]:.4f}", f"{normal[3]:.4f}"])

    def plot_losses(self):
        """绘制训练和验证 Loss 曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Total Train Loss', marker='o')
        plt.plot(self.epochs, self.train_main_losses, label='Main Loss', linestyle='--')
        plt.plot(self.epochs, self.train_aux_losses, label='Aux Loss', linestyle=':')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'loss_plot_{self.timestamp}.png'))
        plt.close()

    def plot_metrics(self):
        """绘制整体、肿瘤样本、正常样本的指标曲线"""
        plt.figure(figsize=(15, 12))

        # --- 整体 ---
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(self.epochs, self.val_dice, label='Dice', color='blue')
        ax1.plot(self.epochs, self.val_precision, label='Precision', color='green')
        ax1.plot(self.epochs, self.val_recall, label='Recall', color='red')
        ax1.plot(self.epochs, self.val_iou, label='IoU', color='purple')
        ax1.set_title('Overall Metrics')
        ax1.grid(True)
        ax1.legend(fontsize=8)

        # --- 肿瘤样本 ---
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(self.epochs, self.val_dice_tumor, label='Dice', color='blue')
        ax2.plot(self.epochs, self.val_precision_tumor, label='Precision', color='green')
        ax2.plot(self.epochs, self.val_recall_tumor, label='Recall', color='red')
        ax2.plot(self.epochs, self.val_iou_tumor, label='IoU', color='purple')
        ax2.set_title('Tumor Sample Metrics')
        ax2.grid(True)
        ax2.legend(fontsize=8)

        # --- 正常样本 ---
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(self.epochs, self.val_dice_normal, label='Dice', color='blue')
        ax3.plot(self.epochs, self.val_precision_normal, label='Precision', color='green')
        ax3.plot(self.epochs, self.val_recall_normal, label='Recall', color='red')
        ax3.plot(self.epochs, self.val_iou_normal, label='IoU', color='purple')
        ax3.set_title('Normal Sample Metrics')
        ax3.grid(True)
        ax3.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'metrics_plot_{self.timestamp}.png'))
        plt.close()



def append_csv_averages(csv_path):
    import pandas as pd
    
    # 读取 CSV 并跳过注释行
    df = pd.read_csv(csv_path, comment='#')
    
    # 只保留数值列
    numeric_cols = [
        'Train Loss', 'Val Loss',
        'Dice', 'Precision', 'Recall', 'IoU',
        'Dice(Tumor)', 'Precision(Tumor)', 'Recall(Tumor)', 'IoU(Tumor)',
        'Dice(Normal)', 'Precision(Normal)', 'Recall(Normal)', 'IoU(Normal)',
        'Time(s)'
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    # 转换为浮点数并计算均值
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    means = df[numeric_cols].mean().fillna(0)
    
    # 写入文件
    with open(csv_path, 'a', newline='') as f:
        f.write("\ --- Averages ---")
        f.write(",".join([""] * 3))
        f.write(",".join(f"{means[col]:.4f}" for col in numeric_cols))
        f.write(",\n")
