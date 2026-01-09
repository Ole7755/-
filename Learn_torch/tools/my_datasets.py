import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class RMBDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
        __init__ 的 Docstring
        rmb面额分类任务数据集
        :param self: 说明
        :param data_dir: str,数据集所在的目录
        :param transform: 数据预处理方式
        """
        self.label_name = {"1": 0, "100": 1}
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    #  在 Python 中，如果一个类里的方法不需要用到对象本身的状态（self），按照规范，
    # 我们通常就把它声明为静态方法。这就相当于告诉阅读代码的人：“这是一个独立的工具函数，
    # 虽然我把它放在这个类里，但它不依赖这个类的具体对象也能工作。”
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = [f for f in img_names if f.endswith(".jpg")]

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info
