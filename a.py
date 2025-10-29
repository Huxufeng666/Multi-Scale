
import torch
from model import FullModel,  DFSANet





if __name__ == "__main__":
    # 创建随机输入
    # x = torch.randn(2, 1, 256, 256)  # batch=2, 单通道, 256x256
    
    # # 创建模型（是否启用 decouple 模块）
    # model = FullModel()
    
    # # 前向传播测试
    # with torch.no_grad():
    #     y,loss = model(x)
    
    #     # print("✅ Forward success!")
            
    #     # print(f"Input shape:  {x.shape}")
    #     print(f"Output shape: {y.shape}")


    # 创建随机输入
    x = torch.randn(2, 1, 256, 256)  # batch=2, 3通道, 256x256    
    gt = torch.randn(2, 1, 256, 256) # 对应的 GT 掩码
    # 创建模型
    model = DFSANet(n_channels=1, n_classes=1, base_c=64)
    # 运行
    model.train() # 切换到训练模式以启用损失计算
    outputs, aux_loss = model(x, gt)
    # 检查输出
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Auxiliary Loss: {aux_loss.item():.4f}")       



