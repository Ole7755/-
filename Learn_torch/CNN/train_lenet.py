import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm

# 环境与配置准备

# 获取当前脚本所在目录 (Week3/) 和 项目根目录 (Learn_torch/)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
sys.path.append(str(PROJECT_DIR))  # 将根目录加入 Python 路径，以便导入 tools

# 导入自定义模块
from tools.my_datasets import RMBDataset
from tools.common_tools import set_seed
from tools.config import load_config
from Week3.lenet import LeNet

# 加载配置
cfg = load_config()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
set_seed(cfg["train"]["seed"])
# 准备 TensorBoard 日志目录 (放到根目录下的 runs 文件夹)
log_dir = PROJECT_DIR / "runs" / f"{cfg['model']['name']}_{cfg['project_name']}"
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(log_dir))

# ============================ step 1/5 数据 ============================
data_root = PROJECT_DIR / cfg["data"]["root_dir"]
train_dir = data_root / "train"
valid_dir = data_root / "valid"

# 定义 Transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)
valid_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)
# 构建 Dataset
train_data = RMBDataset(data_dir=str(train_dir), transform=train_transform)
valid_data = RMBDataset(data_dir=str(valid_dir), transform=valid_transform)

# 构建 DataLoader
train_loader = DataLoader(
    dataset=train_data,
    batch_size=cfg["data"]["train_batch_size"],
    shuffle=True,
    num_workers=cfg["data"]["num_workers"],
)
valid_loader = DataLoader(
    dataset=valid_data,
    batch_size=cfg["data"]["valid_batch_size"],
    num_workers=cfg["data"]["num_workers"],
)

# ============================ step 2/5 模型 ============================
net = LeNet(cfg["model"]["num_classes"])
net.initialize_weights()
net.to(device=device)

# ============================ step 3/4损失函数 & 优化器============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    params=net.parameters(), lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"]
)
# 学习率调整策略：每过 10 个 epoch，学习率乘以 0.1
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
# ============================ step 4/4 训练循环 ============================
# 训练 -> 记录 -> 验证
print(f"Start training {cfg['model']['name']}...")
MAX_EPOCH = cfg["train"]["max_epoch"]
VAL_INTERVAL = cfg["train"]["val_interval"]
best_acc = 0.0
# 创建保存模型的文件夹
save_dir = PROJECT_DIR / "checkpoint"
save_dir.mkdir(exist_ok=True)
for epoch in range(MAX_EPOCH):
    loss_mean = 0.0
    correct = 0.0
    total = 0.0

    net.train()
    # 使用 tqdm 包装 loader 实现进度条
    # file=sys.stdout：指定输出流。sys.stdout 表示标准输出（控制台）
    train_bar = tqdm(train_loader, file=sys.stdout)

    for i, data in enumerate(train_bar):
        inputs, labels = data

        # 将数据搬到 device
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. 前向传播
        outputs = net(inputs)

        # 2. 反向传播
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 3. 统计指标
        # tensor.data 返回的张量不参与梯度计算
        _, predicted = torch.max(outputs.data, 1)
        # tensor.item() 用于将单元素张量转换为 Python 标量值。
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        loss_mean += loss.item()

        # 4. 打印与记录
        # Epoch 1/10:  50%|█████     | 100/200 [loss=0.45, acc=0.82]
        # set_description 是左侧的“任务标题”
        train_bar.set_description(f"Epoch {epoch+1}/{MAX_EPOCH}")
        # set_postfix 的作用是在进度条的右侧动态显示指定的数值
        train_bar.set_postfix(loss=loss.item(), acc=correct / total)

        # 记录每个 Step 的 Training Loss 到 TensorBoard
        writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_loader) + i)

    scheduler.step()

    # ============================ 验证 (Validation) ============================

    if (epoch + 1) % VAL_INTERVAL == 0:
        correct_val = 0.0
        total_val = 0.0
        loss_val = 0.0

        net.eval()
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                loss_val += loss.item()

        val_acc = correct_val / total_val
        val_loss = loss_val / len(valid_loader)

        print(
            f"\n[Epoch {epoch+1}] Valid Acc: {val_acc:.2%} | Valid Loss: {val_loss:.4f}"
        )

        # 记录到 TensorBoard
        writer.add_scalar("Loss/Valid", val_loss, epoch)
        writer.add_scalar("Acc/Valid", val_acc, epoch)

        # === 保存最佳模型 ===
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = (
                save_dir / f"{cfg['model']['name']}_{cfg['project_name']}_best.pth"
            )

            # 保存模型参数
            torch.save(net.state_dict(), str(save_path))
            print(f"Saved Best Model with Acc {best_acc:.2%} to {save_path}")

writer.close()
print("Training Finished!")
