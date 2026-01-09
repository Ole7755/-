import yaml
from pathlib import Path


# 读取YAML配置文件
def load_config(config_path=None):
    """
    加载配置文件。
    如果未指定路径，则默认寻找项目根目录下的 config.yaml
    """
    if config_path is None:
        # typt(root_dir): pathlib.PosixPath
        root_dir = Path(__file__).resolve().parents[1]
        config_path = root_dir / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:

        cfg = yaml.safe_load(f)

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(type(cfg))
    print(cfg["model"]["name"])
