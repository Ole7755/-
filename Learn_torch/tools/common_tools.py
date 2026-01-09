import torch
import random
import psutil
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def set_seed(seed=1):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.mps.manual_seed(1)


def transform_invert(img_: torch.Tensor, transform_train):
    """
    将data 进行反transfrom操作

    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if "Normalize" in str(transform_train):
        norm_transform = list(
            filter(
                lambda x: isinstance(x, transforms.Normalize),
                transform_train.transforms,
            )
        )
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.permute(1, 2, 0)

    if "ToTensor" in str(transform_train) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255
    # 处理彩色图像 (3 通道)
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype("uint8")).convert("RGB")
    # 处理灰度图像 (1 通道)
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype("uint8").squeeze())
    else:
        raise Exception(
            "Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(
                img_.shape[2]
            )
        )

    return img_
