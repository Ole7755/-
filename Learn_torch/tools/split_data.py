import os
import random
import shutil
from glob import glob
from pathlib import Path


def main():
    # 配置
    CURRENT_DIR = Path(__file__).resolve()
    WORK_DIR = CURRENT_DIR.parent.parent
    DATASET_DIR = WORK_DIR / "data" / "imagenet-10"
    output_root = WORK_DIR / "data" / "imagenet-10-split"
    output_root.mkdir(parents=True, exist_ok=True)
    split_ratio = 0.2

    classes = [cls.name for cls in DATASET_DIR.iterdir() if cls.is_dir()]

    train_path = output_root / "train"
    train_path.mkdir(exist_ok=True)
    test_path = output_root / "test"
    test_path.mkdir(exist_ok=True)

    for cls in classes:
        cls_root = DATASET_DIR / cls
        train_cls = train_path / cls
        train_cls.mkdir(exist_ok=True)
        test_cls = test_path / cls
        test_cls.mkdir(exist_ok=True)

        imgs = [f.name for f in cls_root.iterdir() if f.is_file()]
        num = len(imgs)

        random.shuffle(imgs)
        test_index = int(num * split_ratio)

        test_imgs = imgs[:test_index]
        train_imgs = imgs[test_index:]

        for img in test_imgs:
            dest_img = test_cls / img
            source_img = cls_root / img
            shutil.copy(source_img, dest_img)
        for img in train_imgs:
            dest_img = train_cls / img
            source_img = cls_root / img
            shutil.copy(source_img, dest_img)


if __name__ == "__main__":
    main()
