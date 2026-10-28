# -*- coding: utf-8 -*-
"""
猫狗分割 (高亮动物) —— 使用预训练 DeepLabV3
==========================================
数据没有分割掩码, 因此无法从零训练分割模型.
这里的做法是使用 torchvision 自带的、已在 Pascal VOC 上预训练好的
DeepLabV3 模型 (该模型本身就能识别 cat / dog), 对图片做像素级分割,
把动物区域用彩色高亮出来, 并标注是 cat 还是 dog.

VOC 类别编号: background=0, ..., cat=8, ..., dog=12
"""

import os

import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "train/train"
RESULT_DIR = os.path.join("results", "segmentation")
WORK_SIZE = 520          # 送入模型的工作分辨率
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 只关心这两个类别
CAT_ID, DOG_ID = 8, 12
LABEL = {CAT_ID: "cat", DOG_ID: "dog"}
COLOR = {CAT_ID: (0, 200, 0), DOG_ID: (255, 128, 0)}   # 猫绿色, 狗橙色


def load_model():
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=weights)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((WORK_SIZE, WORK_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def overlay_mask(rgb_img, mask, class_id):
    """把布尔 mask 用对应颜色半透明地叠加到原图上."""
    arr = np.array(rgb_img).astype(np.float32)
    color = np.array(COLOR.get(class_id, (150, 150, 150)), dtype=np.float32)
    alpha = 0.45
    for c in range(3):
        arr[:, :, c] = np.where(mask, arr[:, :, c] * (1 - alpha) + color[c] * alpha, arr[:, :, c])
    return Image.fromarray(arr.astype(np.uint8))


def draw_label(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    # 文本背景条
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([5, 5, 5 + w + 16, 5 + h + 12], fill=(0, 0, 0))
    draw.text((13, 11), text, fill=(255, 255, 255), font=font)
    return img


def segment_one(model, transform, path):
    """对单张图片做分割, 返回 (叠加后图片, 标签文本)."""
    orig = Image.open(path).convert("RGB")
    work = orig.resize((WORK_SIZE, WORK_SIZE))
    x = transform(work).unsqueeze(0)
    with torch.no_grad():
        out = model(x)["out"]           # [1, 21, 520, 520]
    pred = out.argmax(dim=1)[0].numpy() # [520, 520]

    # 统计该图中是猫还是狗的像素最多
    counts = {cid: int((pred == cid).sum()) for cid in (CAT_ID, DOG_ID)}
    class_id = max(counts, key=counts.get)
    mask = pred == class_id

    # 若几乎没有任何动物像素, 视为未检出
    if counts[class_id] < 200:
        label = "no animal"
        class_id = None
    else:
        label = LABEL[class_id]

    colored = overlay_mask(work, mask, class_id)
    colored = draw_label(colored, label)
    return orig, colored, label


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("加载预训练 DeepLabV3 模型 (首次会自动下载权重) ...")
    model = load_model()
    transform = get_transform()

    # 选择 10 张示例图片: 5 猫 + 5 狗
    samples = [
        os.path.join(DATA_DIR, "cat.0.jpg"),
        os.path.join(DATA_DIR, "cat.10.jpg"),
        os.path.join(DATA_DIR, "cat.123.jpg"),
        os.path.join(DATA_DIR, "cat.250.jpg"),
        os.path.join(DATA_DIR, "cat.2500.jpg"),
        os.path.join(DATA_DIR, "dog.0.jpg"),
        os.path.join(DATA_DIR, "dog.10.jpg"),
        os.path.join(DATA_DIR, "dog.123.jpg"),
        os.path.join(DATA_DIR, "dog.250.jpg"),
        os.path.join(DATA_DIR, "dog.2500.jpg"),
    ]

    for s in samples:
        if not os.path.exists(s):
            print(f"跳过不存在: {s}")
            continue
        orig, colored, label = segment_one(model, transform, s)
        name = os.path.splitext(os.path.basename(s))[0]
        # 左右拼接: 原图 | 分割结果
        side = Image.new("RGB", (WORK_SIZE * 2, WORK_SIZE))
        side.paste(orig.resize((WORK_SIZE, WORK_SIZE)), (0, 0))
        side.paste(colored, (WORK_SIZE, 0))
        save_path = os.path.join(RESULT_DIR, f"{name}_seg.png")
        side.save(save_path)
        print(f"  {name}: 标签 = {label}  ->  {save_path}")

    print(f"\n分割结果已保存到 {RESULT_DIR}/")


if __name__ == "__main__":
    main()