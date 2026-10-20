# 导入核心深度学习库
import torch  # 张量计算和自动微分
import torch.nn as nn  # 神经网络模块
from torch.utils.data import Dataset, DataLoader  # 自定义数据集和批量加载
from torchvision import transforms  # 图像预处理
from PIL import Image  # 图像处理库
import pandas as pd  # 数据表格处理
import json  # 配置文件解析
from pathlib import Path  # 路径操作
import os  # 文件操作
import numpy as np  # 数值计算
import shutil  # 文件操作

# 自定义数据集类，继承torch.utils.data.Dataset
class CNNDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        # 从CSV文件加载数据元信息
        self.df = pd.read_csv(csv_path)  # 包含图像路径和标签
        self.img_dir = Path(img_dir)  # 图像存储根目录
        # 定义数据预处理流水线，默认包含：
        self.transform = transform or transforms.Compose([
            transforms.Resize((28, 28)),  # 统一图像尺寸为28x28（MNIST标准）
            transforms.ToTensor(),        # 转换为PyTorch张量
            transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理（MNIST均值方差）
        ])
    
    # 返回数据集样本数量
    def __len__(self):
        return len(self.df)
    
    # 按索引获取单个样本
    def __getitem__(self, idx):
        # 读取CSV中的元数据
        row = self.df.iloc[idx]
        # 构建图像完整路径
        img_path = self.img_dir / row['filename']
        # 打开图像并转换为RGB格式（兼容灰度图）
        image = Image.open(img_path).convert('RGB')
        # 获取标签并转换为整数
        label = int(row['label'])
        # 应用预处理并返回样本
        return self.transform(image), label

# 定义CNN模型架构
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取模块（卷积层+池化层）
        self.features = nn.Sequential(
            # 输入通道3（RGB），输出通道32，3x3卷积核，保持空间分辨率
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),  # 引入非线性
            nn.MaxPool2d(2, 2),  # 下采样，尺寸减半
            
            # 第二个卷积块
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 分类器模块（全连接层）
        self.classifier = nn.Sequential(
            # 展平后输入维度：64 * 7 * 7（经过两次池化）
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(128, num_classes)  # 输出层
        )
    
    # 前向传播定义
    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = torch.flatten(x, 1)  # 展平操作（batch维度保留）
        return self.classifier(x)  # 分类决策

# 主训练流程
def main():
    print("Starting CNN training...")
    
    # 1. 读取配置文件
    config_path = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(__file__), 'config.json'))
    with open(config_path) as f:
        cfg = json.load(f)
    
    # 2. 设备配置（CPU/GPU根据配置选择）
    use_gpu = cfg['cnn'].get('use_gpu', False)
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if use_gpu:
            print("GPU requested but not available, using CPU instead")
        else:
            print("Using CPU")
    
    # 3. 数据加载与预处理
    data_dir = os.getenv("DATA_DIR", cfg['cnn']['data_dir'])
    print(f"Using data directory: {data_dir}")
    
    # 训练集
    train_dataset = CNNDataset(
        csv_path=os.path.join(data_dir, 'train.csv'),
        img_dir=os.path.join(data_dir, 'train-images')
    )
    # 测试集
    test_dataset = CNNDataset(
        csv_path=os.path.join(data_dir, 'test.csv'),
        img_dir=os.path.join(data_dir, 'test-images')
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['cnn']['batch_size'],
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['cnn']['batch_size'],
        shuffle=False
    )
    
    # 4. 模型初始化
    model = SimpleCNN(num_classes=cfg['cnn']['classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['cnn']['lr'])
    criterion = nn.CrossEntropyLoss()
    
    print("Training started...")
    # 5. 训练循环
    for epoch in range(cfg['cnn']['epochs']):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        # 批量处理数据
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} processed")
        
        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # 打印训练进度
        print(f"Epoch {epoch+1}/{cfg['cnn']['epochs']}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        # 验证阶段
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += len(target)
        
        test_avg_loss = test_loss / len(test_loader)
        test_accuracy = 100. * test_correct / test_total
        print(f"          Test Loss={test_avg_loss:.4f}, Test Acc={test_accuracy:.2f}%")
    
    # 6. 模型保存
    model_path = "cnn_model_full.pt"
    torch.save(model, model_path)
    print(f"CNN training completed! Model saved as {model_path}")

if __name__ == "__main__":
    main()
