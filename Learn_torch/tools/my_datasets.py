import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(root_dir, batch_size, is_train=True, num_workers=0):
    """
    构建 DataLoader
    :param root_dir: 数据集根目录 (e.g., 'data/imagenet-10-split/train')
    :param batch_size: 批大小
    :param is_train: 是否为训练集（决定是否应用增强）
    :param num_workers: 加载线程数
    """

    # 1. 定义 Transform
    # ImageNet 标准归一化参数
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
