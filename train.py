# -*- coding: utf-8 -*-
"""
猫狗分类 —— 自建小型卷积神经网络 (CNN)
====================================
数据集: Kaggle Dogs vs Cats (train/train 下 cat.*.jpg 与 dog.*.jpg)
流程:
    1. 读取图片并按文件名前缀得到猫/狗标签
    2. 划分训练集(每类 TRAIN_PER_CLASS) 与 测试集(每类 TEST_PER_CLASS)
    3. 定义一个小型 CNN 网络
    4. 训练, 每个 epoch 打印训练损失与测试准确率
    5. 保存: 模型权重 + 训练曲线 + 混淆矩阵 + 抽样预测图
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # 无界面环境下也能保存图片
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------------------- 超参数 -----------------------------
DATA_DIR = "train/train"      # 原始图片目录
IMG_SIZE = 128                 # 统一缩放到 128x128
TRAIN_PER_CLASS = 3000         # 训练集: 每类图片数
TEST_PER_CLASS = 500           # 测试集: 每类图片数
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
LR_STEP = 5                    # 每隔多少 epoch 降低学习率
LR_GAMMA = 0.5                 # 学习率衰减系数
WEIGHT_DECAY = 1e-4
SEED = 42
RESULT_DIR = "results"

CLASSES = ["cat", "dog"]        # 0 -> cat, 1 -> dog


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_paths(data_dir):
    """按是否为损坏图片过滤, 返回 (猫路径列表, 狗路径列表)."""
    cats, dogs = [], []
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(".jpg"):
            continue
        path = os.path.join(data_dir, name)
        # 跳过无法读取的损坏图片
        try:
            with Image.open(path) as im:
                im.load()
        except Exception:
            continue
        if name.lower().startswith("cat"):
            cats.append(path)
        elif name.lower().startswith("dog"):
            dogs.append(path)
    return cats, dogs


def split_data(cats, dogs):
    """随机抽取训练/测试样本, 保证两类数量相同. 返回 (train_items, test_items)."""
    random.shuffle(cats)
    random.shuffle(dogs)
    train = [(p, 0) for p in cats[:TRAIN_PER_CLASS]] + \
            [(p, 1) for p in dogs[:TRAIN_PER_CLASS]]
    test = [(p, 0) for p in cats[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]] + \
           [(p, 1) for p in dogs[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]]
    random.shuffle(train)
    random.shuffle(test)
    return train, test


class CatDogDataset(Dataset):
    """简单的数据集类: 给定 (路径, 标签) 列表, 返回 (图片张量, 标签)."""

    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ----------------------------- 网络结构 -----------------------------
class CNN(nn.Module):
    """小型 CNN: 4 个卷积块 + 2 层全连接. 结构直观, 便于讲解."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),    # 32x64x64
            self._block(32, 64),   # 64x32x32
            self._block(64, 128),  # 128x16x16
            self._block(128, 128), # 128x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def make_loader(items, train):
    if train:
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),          # 随机水平翻转
            transforms.RandomRotation(10),              # 随机小幅旋转
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    ds = CatDogDataset(items, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def plot_curves(train_losses, test_accs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(range(1, len(test_accs) + 1), test_accs, marker="o", color="green")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_samples(model, items, save_path, n=8):
    """展示若干测试图片的预测结果 (真实标签 vs 预测标签)."""
    model.eval()
    idxs = np.random.choice(len(items), size=min(n, len(items)), replace=False)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for ax, i in zip(axes, idxs):
        path, y = items[i]
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0)
        pred = model(x).argmax(dim=1).item()
        # 反归一化用于显示
        show = tf(img).clamp(-1, 1) * 0.5 + 0.5
        ax.imshow(show.permute(1, 2, 0))
        color = "green" if pred == y else "red"
        ax.set_title(f"true:{CLASSES[y]}\npred:{CLASSES[pred]}", color=color, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    set_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("1) 读取并划分数据 ...")
    cats, dogs = collect_paths(DATA_DIR)
    print(f"   有效图片: cat={len(cats)}  dog={len(dogs)}")
    train_items, test_items = split_data(cats, dogs)
    print(f"   训练集 {len(train_items)} 张, 测试集 {len(test_items)} 张")

    train_loader = make_loader(train_items, train=True)
    test_loader = make_loader(test_items, train=False)

    model = CNN(num_classes=2)
    print(f"\n2) 网络参数量: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n3) 开始训练 (CPU)...\n")
    train_losses, test_accs = [], []
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, opt, loss_fn)
        acc = evaluate(model, test_loader)
        train_losses.append(loss)
        test_accs.append(acc)
        scheduler.step()
        print(f"   Epoch {epoch:2d}/{EPOCHS}   loss={loss:.4f}   test_acc={acc:.4f}   lr={scheduler.get_last_lr()[0]:.2e}")

    # 最终测试结果
    final_acc = test_accs[-1]
    print(f"\n4) 最终测试准确率: {final_acc:.4f} ({final_acc * 100:.2f}%)")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "catdog_cnn.pth"))

    # 混淆矩阵
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            y_true.extend(y.tolist())
            y_pred.extend(model(x).argmax(dim=1).tolist())
    print(f"   混淆矩阵准确率: {accuracy_score(y_true, y_pred):.4f}")

    # 绘图
    plot_curves(train_losses, test_accs, os.path.join(RESULT_DIR, "training_curves.png"))
    plot_confusion(y_true, y_pred, os.path.join(RESULT_DIR, "confusion_matrix.png"))
    plot_samples(model, test_items, os.path.join(RESULT_DIR, "sample_predictions.png"))
    print(f"\n5) 结果已保存到 {RESULT_DIR}/  (模型 + 3 张结果图)")


if __name__ == "__main__":
    main()