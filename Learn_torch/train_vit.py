import os
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Models.ViT import VisionTransformer, VitConfig
from tools.my_datasets import get_dataloader
from tqdm import tqdm
from pathlib import Path


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": running_loss / total, "acc": correct / total})

    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": running_loss / total, "acc": correct / total})

    return running_loss / len(dataloader), correct / total


def main():
    cfg = load_config("config/config_vit.yaml")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_loader = get_dataloader(
        cfg["data"]["train_root"],
        cfg["data"]["patch_size"],
        True,
        cfg["data"]["num_workers"],
    )
    val_loader = get_dataloader(
        cfg["data"]["val_root"],
        cfg["data"]["batch_size"],
        is_train=False,
        num_workers=cfg["data"]["num_workers"],
    )
