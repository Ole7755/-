import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    ResNet 的基本残差块 (Basic Block)。
    用于 ResNet-18 和 ResNet-34。
    结构：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (Shortcut) -> ReLU
    """
    def __init__(self, in_channel, out_channel, stride, shortcut=None):
        super().__init__()
        # 记录参数，虽然成员变量不是必须的，但方便调试
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.shortcut = shortcut

        # 定义主路径 (Main Path) 的卷积层序列
        self.block = nn.Sequential(
            # 第一层卷积：可能进行下采样 (stride > 1)
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.out_channel,
                kernel_size=3,
                stride=self.stride, # 这里的 stride 决定了是否改变特征图尺寸
                padding=1,          # padding=1 配合 kernel_size=3 保持尺寸不变(当stride=1时)
                bias=False,         # 后面接了 BN，所以不需要 bias
            ),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True), # inplace=True 节省内存

            # 第二层卷积：不进行下采样 (stride 总是 1)
            nn.Conv2d(
                in_channels=self.out_channel,
                out_channels=self.out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channel),
        )

    def forward(self, x):
        # 1. 计算主路径的输出 F(x)
        out = self.block(x)

        # 2. 计算快捷连接 (Shortcut) 的输出
        # 如果 self.shortcut 为 None，直接使用 x；否则通过 shortcut 层调整 x 的维度
        residual = x if self.shortcut is None else self.shortcut(x)

        # 3. 残差连接：F(x) + x
        out += residual

        # 4. 最后经过 ReLU 激活
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 主体架构。
    默认参数实现 ResNet-18。
    """
    def __init__(self, num_classes=10, layer_blocks=[2, 2, 2, 2]):
        super().__init__()
        # 追踪当前的输入通道数，初始为 64 (conv1 的输出)
        self.in_channel = 64

        # ----------------------------------------
        # 1. 预处理层 (Stem Layer)
        # ----------------------------------------
        # 输入: (3, 224, 224) -> 输出: (64, 112, 112)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        # 输入: (64, 112, 112) -> 输出: (64, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ----------------------------------------
        # 2. 残差层 (Residual Layers)
        # ----------------------------------------
        # layer1: 对应 conv2_x，输出通道 64，不进行下采样
        # 尺寸: (64, 56, 56) -> (64, 56, 56)
        self.layer1 = self._make_layer(ResidualBlock, 64, layer_blocks[0], stride=1)

        # layer2: 对应 conv3_x，输出通道 128，下采样
        # 尺寸: (64, 56, 56) -> (128, 28, 28)
        self.layer2 = self._make_layer(ResidualBlock, 128, layer_blocks[1], stride=2)

        # layer3: 对应 conv4_x，输出通道 256，下采样
        # 尺寸: (128, 28, 28) -> (256, 14, 14)
        self.layer3 = self._make_layer(ResidualBlock, 256, layer_blocks[2], stride=2)

        # layer4: 对应 conv5_x，输出通道 512，下采样
        # 尺寸: (256, 14, 14) -> (512, 7, 7)
        self.layer4 = self._make_layer(ResidualBlock, 512, layer_blocks[3], stride=2)

        # ----------------------------------------
        # 3. 分类头 (Classification Head)
        # ----------------------------------------
        # 自适应平均池化，无论输入特征图多大，都输出 (512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层：512 -> num_classes
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 预处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 经过 4 个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平: (Batch, 512, 1, 1) -> (Batch, 512)
        x = self.fc(x)

        return x

    def _make_layer(
        self, block: ResidualBlock, channel: int, block_num: int, stride: int
    ):
        """
        构建一个残差层 (Layer)，包含多个 ResidualBlock。
        :param block: 残差块类 (ResidualBlock)
        :param channel: 该层的目标输出通道数
        :param block_num: 该层包含的 block 数量
        :param stride: 该层第一个 block 的步长
        """
        shortcut = None

        # 判断是否需要下采样或通道数调整
        # 1. stride != 1: 说明尺寸会缩小，Shortcut 需要用 1x1 卷积调整尺寸
        # 2. self.in_channel != channel: 说明通道数改变，Shortcut 需要用 1x1 卷积调整通道
        if stride != 1 or self.in_channel != channel:
            shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel,
                    out_channels=channel,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=channel),
            )

        layers = []
        # 第一个 Block：处理 stride 和 shortcut
        layers.append(block(self.in_channel, channel, stride, shortcut))

        # 更新 self.in_channel 为当前层的输出通道数
        self.in_channel = channel

        # 后续 Block：stride 总是 1，不需要 shortcut
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, stride=1, shortcut=None))

        return nn.Sequential(*layers)