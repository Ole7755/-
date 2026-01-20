import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VitConfig:
    """
    ViT 模型配置类
    用于集中管理模型的超参数，方便统一修改和实验管理。

    :param num_classes: 分类任务的类别数量 (例如: CIFAR-10为10, ImageNet为1000)
    :param img_size: 输入图像的尺寸 (假设高宽相等，例如 224)
    :param patch_size: 将图像切分为小块的大小 (例如 16x16)
    :param in_chans: 输入图像的通道数 (RGB图像为3)
    :param embed_dim: Transformer 内部特征向量的维度 (Embedding Dimension)
    :param depth: Transformer Encoder Block 的层数 (深度)
    :param num_heads: 多头注意力机制 (Multi-Head Attention) 的头数
    :param mlp_ratio: MLP 层中隐藏层的膨胀比率 (通常为 4.0，即 hidden_dim = 4 * embed_dim)
    :param drop_rate: 线性层后的 Dropout 概率
    :param attn_drop_rate: 注意力矩阵 (Attention Map) 的 Dropout 概率
    """

    num_classes: int = 10
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    将二维图像切分为 patch，并映射为一维向量序列。
    实现方式：使用卷积核大小和步长都等于 patch_size 的卷积层。
    """

    def __init__(self, config: VitConfig):
        super().__init__()
        self.img_size = config.img_size
        self.in_channels = config.in_chans
        self.out_channels = config.embed_dim
        self.kernel_size = config.patch_size
        self.stride = config.patch_size

        # 使用卷积层一次性完成切片和线性映射
        # 输入: (B, 3, H, W)
        # 输出: (B, embed_dim, H/P, W/P)
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # 校验输入尺寸，避免因尺寸不匹配导致的潜在错误
        assert H == self.img_size, f"输入图片高度 {H} 与配置 {self.img_size} 不符"
        assert W == self.img_size, f"输入图片宽度 {W} 与配置 {self.img_size} 不符"

        # 1. 卷积投影
        # x: (B, C, H, W) -> (B, embed_dim, H_patch, W_patch)
        x = self.conv(x)

        # 2. 展平 (Flatten)
        # 将 H_patch * W_patch 展平为序列长度 N
        # x: (B, embed_dim, H_patch, W_patch) -> (B, embed_dim, N)
        # 其中 N = (img_size // patch_size) ** 2
        x = x.view(B, self.out_channels, -1).contiguous()

        # 3. 维度置换 (Transpose)
        # Transformer 需要的输入格式是 (Batch, Sequence_Length, Embedding_Dim)
        # x: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.permute(0, 2, 1)

        return x


class Mlp(nn.Module):
    """
    多层感知机 (Multi-Layer Perceptron)
    Transformer Block 中的前馈网络部分。
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, config: VitConfig):
        super().__init__()
        # 隐藏层维度通常是嵌入维度的 4 倍
        hidden_features = int(config.embed_dim * config.mlp_ratio)

        self.linear1 = nn.Linear(config.embed_dim, hidden_features)
        self.gelu = nn.GELU()  # GELU 是 Transformer 中标准的激活函数
        self.linear2 = nn.Linear(hidden_features, config.embed_dim)
        self.dropout = nn.Dropout(p=config.drop_rate)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention, MSA)
    ViT 的核心组件，用于捕捉序列中 patch 之间的全局关系。
    """

    def __init__(self, config: VitConfig):
        super().__init__()
        # 确保嵌入维度可以被头数整除
        assert (
            config.embed_dim % config.num_heads == 0
        ), "embed_dim 必须能被 num_heads 整除"

        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        # 缩放因子 (Scale Factor): 1 / sqrt(head_dim)
        # 防止点积结果过大导致 Softmax 梯度消失
        self.scale = self.head_dim**-0.5

        # qkv: 一次性计算 Query, Key, Value
        # 输出维度是 embed_dim * 3
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3, bias=True)

        # Attention Map 的 Dropout
        self.attn_drop = nn.Dropout(config.attn_drop_rate)

        # 输出投影层
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        # 输出层的 Dropout
        self.proj_drop = nn.Dropout(config.drop_rate)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        # C 应该等于 embed_dim

        # 1. 计算 Q, K, V
        # self.qkv(x): (B, N, 3 * C)
        # reshape: (B, N, 3, num_heads, head_dim)
        # permute: (3, B, num_heads, N, head_dim) -> 把 3 移到最前面方便解包，把 num_heads 移到前面方便并行计算
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 (B, num_heads, N, head_dim)

        # 2. 计算 Attention Scores
        # attn = (Q @ K^T) * scale
        # q: (B, num_heads, N, head_dim)
        # k.transpose: (B, num_heads, head_dim, N)
        # @: 矩阵乘法 -> (B, num_heads, N, N) -> 得到 N 个 patch 之间的两两关系
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. 加权求和
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 还原回 (B, N, C)

        # 4. 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Encoder Block
    包含: Norm -> Attention -> Residual -> Norm -> MLP -> Residual
    """

    def __init__(self, config: VitConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = Mlp(config=config)

    def forward(self, x):
        # Pre-Norm 结构: 先 Norm 再 Attention/MLP
        # Residual Connection: 输入 + 输出
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 主模型类
    论文: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    """

    def __init__(self, config: VitConfig):
        super().__init__()
        # 1. Patch Embedding 层
        self.patch_embed = PatchEmbed(config)

        # 2. Class Token (可学习参数)
        # 类似于 BERT 中的 [CLS] token，用于汇总全局图像信息进行分类
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # 3. Positional Embedding (可学习参数)
        # 用于给每个 patch 添加位置信息
        # 长度 = patch 数量 + 1 (cls_token)
        num_patches = (config.img_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.drop_rate)

        # 4. Transformer Encoder Blocks 堆叠
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.depth)])

        # 5. 最终归一化层和分类头
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        # 6. 初始化权重
        self.apply(self._init_weights)
        # 单独初始化 pos_embed 和 cls_token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_weights(self, m):
        """
        参数初始化策略:
        - Linear/Conv2d: 截断正态分布 (std=0.02)
        - LayerNorm: bias=0, weight=1
        """
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor):
        """
        特征提取过程
        """
        # (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x)

        # 扩展 cls_token 并拼接到输入序列的最前面
        # cls_token: (1, 1, D) -> (B, 1, D)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # cat: (B, 1, D) + (B, N, D) -> (B, N+1, D)
        x = torch.cat((cls_token, x), dim=1)

        # 加上位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过所有 Transformer Block
        for block in self.blocks:
            x = block(x)

        # 最终归一化
        x = self.norm(x)

        # 只提取 cls_token 对应的输出 (第 0 个位置) 用于分类
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # 简单的测试代码
    torch.manual_seed(42)

    # 使用较小的配置进行测试
    config = VitConfig(
        num_classes=10, img_size=224, patch_size=16, embed_dim=384, depth=4, num_heads=6
    )

    model = VisionTransformer(config)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == (1, 10)
    print("Test Passed!")
