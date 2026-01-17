import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VitConfig:
    """
    配置类
    :param num_classes: 类别数
    :param img_size: 输入图片大小
    :param patch_size: Patch 大小
    :param in_chans: 说明
    :param embed_dim: 说明
    :param depth: 说明
    :param num_heads: 说明
    :param mlp_ratio: 说明
    :param drop_rate: 说明
    :param attn_drop_rate: 说明
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
    def __init__(self, config: VitConfig):
        super().__init__()
        self.img_size = config.img_size
        self.in_channels = config.in_chans
        self.out_channels = config.embed_dim
        self.kernel_size = config.patch_size
        self.stride = config.patch_size
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H == self.img_size, "输入图片尺寸和配置文件不一样"
        x = self.conv(x)
        x = x.view(B, self.out_channels, -1).contiguous()
        x = x.permute(0, 2, 1)

        return x


class Mlp(nn.Module):
    def __init__(self, config: VitConfig):
        super().__init__()
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        self.linear1 = nn.Linear(config.embed_dim, hidden_features)
        self.gelu = nn.GELU()
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
    def __init__(self, config: VitConfig):
        super().__init__()


if __name__ == "__main__":
    pass
