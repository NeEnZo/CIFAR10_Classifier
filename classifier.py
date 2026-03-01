# CIFAR-10 改造版（迁移学习 + 强增强 + 验证集）
# 本文件是原始代码的复制改造版，关键改动都用 `# ===== 修改X: ... =====` 标注。

import sys
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import pickle
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
import random
from PIL import Image

# ===== 修改1: 固定随机种子，提升可复现性 =====
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# cuDNN 确定性模式：禁用 benchmark 自动选算法，强制使用固定算法
# 不设置这两行，即使 seed 相同，GPU 卷积结果也会有微小随机浮动
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """DataLoader 子进程种子初始化：num_workers > 0 时每个 worker 是独立进程，
    不会继承主进程的 numpy/random 种子，必须手动设置，否则数据增强的随机性每次不同"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备: {device}')


# ===== 修改2: 使用更强数据增强（RandAugment + RandomErasing）=====
# 为了迁移学习模型（ResNet18 预训练）适配，输入调整到 224x224
transform_train = transforms.Compose([
    transforms.Resize((224, 224)), # 调整输入大小以适配预训练模型
    transforms.RandomCrop(224, padding=16), # 随机裁剪并添加填充
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
    transforms.RandAugment(num_ops=2, magnitude=9), # RandAugment 数据增强，即随机选择两种增强操作，强度为9
    transforms.ToTensor(), # 转换为张量，以适配预训练模型输入
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 标准化，使用 CIFAR-10 数据集的均值和标准差
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)) # 随机擦除，概率为0.25，擦除区域的面积占比在2%到20%之间，宽高比在0.3到3.3之间
])

transform_eval = transforms.Compose([ # 评估时不使用数据增强，只进行必要的预处理
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class MyCIFAR10(Dataset): # 自定义数据集类，继承自 PyTorch 的 Dataset，用于加载 CIFAR-10 数据集
    def __init__(self, root, train=True): # 初始化方法，接受数据集根目录和是否为训练集的标志，加载数据并进行预处理
        self.root = root
        self.train = train
        self.data = []
        self.targets = []

        if self.train:
            files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        else:
            files = ['test_batch']

        for file_name in files:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32) # 将数据堆叠成一个大数组，并调整形状为 (N, 3, 32, 32)，其中 N 是样本数量，3 是通道数，32x32 是图像尺寸
        self.data = self.data.transpose((0, 2, 3, 1)) # 转置数据为 (N, 32, 32, 3)，以适配 PIL 图像格式

    def __getitem__(self, index): # 获取指定索引的数据项，返回图像和对应的标签
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        return img, target

    def __len__(self):
        return len(self.data)


# ===== 修改3: 增加带 transform 的子集包装器，便于划分 train/val 后使用不同变换 =====
class TransformSubset(Dataset): # 一个包装器类，用于在子集上应用特定的变换，接受一个基础数据集、索引列表和变换函数作为输入
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        image, target = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.indices)


# ===== 修改5: 迁移学习模型（ResNet50 预训练）=====
# 升级 ResNet18 → ResNet50：参数量翻倍（25M vs 11M），layer4 输出 2048 维特征
# 更深的网络能提取更精细的局部纹理（耳廓、鼻型、毛色分布），有效缓解猫/狗混淆
def build_model(num_classes=10):
    try:
        weights = models.ResNet50_Weights.DEFAULT # 使用 ResNet50 的默认预训练权重（在 ImageNet 上训练的权重）
        model = models.resnet50(weights=weights) # 加载 ResNet50 模型，并应用预训练权重
        print('使用预训练权重: ResNet50_Weights.DEFAULT')
    except Exception as e:
        print(f'预训练权重加载失败，改为随机初始化: {e}')
        model = models.resnet50(weights=None) # 加载 ResNet50 模型，随机初始化权重

    in_features = model.fc.in_features # 获取 ResNet50 最后全连接层的输入特征数（通常是 2048）
    model.fc = nn.Linear(in_features, num_classes) # 替换最后全连接层，输出维度等于类别数（10）
    return model


def evaluate(net, dataloader, criterion, device):
    # net指向模型，dataloader是数据加载器，criterion是损失函数，device是计算设备（CPU或GPU）
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader: # 遍历数据加载器中的每个批次，获取图像和对应的标签
            images, labels = images.to(device), labels.to(device)
            outputs = net(images) # 将图像输入模型，得到输出（预测结果）
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def test_with_required_classes(net, testloader, device, classes):
    print('开始测试...')
    net.eval()

    class_correct = [0 for _ in range(10)] # 用于统计每个类别的正确预测数量，初始化为 0
    class_total = [0 for _ in range(10)] # 用于统计每个类别的总测试样本数量，初始化为 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            preds = outputs.argmax(dim=1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                class_correct[label] += int(preds[i].item() == label)

    overall_acc = 100.0 * sum(class_correct) / sum(class_total)
    print(f'在整个测试集上的准确率: {overall_acc:.2f}%')

    # ===== 输出全部 10 个类别的准确率 =====
    print('\n--- 全部类别分类准确率 ---')
    print(f'{"类别":<12} {"正确":>6} {"总数":>6} {"准确率":>8}')
    print('-' * 38)
    for i in range(10):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            marker = ' ◀ 考核' if i in [0, 3, 6] else ''  # 标记考核三类
            print(f'{classes[i]:<12} {class_correct[i]:>6} {class_total[i]:>6} {acc:>7.2f}%{marker}')

    print('-' * 38)
    indices_to_test = [0, 3, 6]
    total_correct = sum(class_correct[i] for i in indices_to_test)
    total_tested = sum(class_total[i] for i in indices_to_test)
    if total_tested > 0:
        avg_acc = 100.0 * total_correct / total_tested
        print(f'\n飞机、猫、青蛙三类别的平均准确率: {avg_acc:.2f}%')


# ===== 修改9: CutMix + Mixup 数据增强，提升易混淆类别（猫/狗）的区分度 =====
# CutMix: 将一张图片的矩形区域替换为另一张图片的对应区域，标签按面积比例混合
# Mixup:  将两张图片按比例线性混合，标签也按相同比例混合
# 两者都迫使模型关注局部细粒度特征，而非依赖全局轮廓，有效缓解猫/狗等相似类别的混淆

def rand_bbox(size, lam):
    """根据混合比例 lam 随机生成裁剪框坐标"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """CutMix: 随机用另一张图片的区域替换当前图片的区域"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # 根据实际裁剪面积重新计算 lam
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y, y[index], lam


def mixup_data(x, y, alpha=0.8):
    """Mixup: 将两张图片按 lam 比例线性混合"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    """混合标签的损失计算: loss = lam * L(pred, y_a) + (1-lam) * L(pred, y_b)"""
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)


if __name__ == '__main__':
    # 加载 CIFAR-10 数据集
    data_dir = './2025zjutestfinal'

    # ===== 修改4: 从训练集划分验证集（9:1）=====
    base_trainset = MyCIFAR10(root=data_dir, train=True)
    num_total = len(base_trainset)
    num_val = int(0.1 * num_total)
    num_train = num_total - num_val

    g = torch.Generator().manual_seed(seed) # 使用固定随机种子生成随机索引，确保划分的可复现性
    perm = torch.randperm(num_total, generator=g).tolist() # 生成一个随机排列的索引列表，长度为总样本数
    train_indices = perm[:num_train] # 前 num_train 个索引用于训练集
    val_indices = perm[num_train:] # 后 num_val 个索引用于验证集

    trainset = TransformSubset(base_trainset, train_indices, transform_train) # 使用 TransformSubset 包装训练集，应用训练数据增强变换
    valset = TransformSubset(base_trainset, val_indices, transform_eval)
    test_base = MyCIFAR10(root=data_dir, train=False) # 测试集不需要划分，直接使用原始数据加载器加载测试数据
    testset = TransformSubset(test_base, list(range(len(test_base))), transform_eval)

    # generator 参数控制 DataLoader 的 shuffle 随机性，worker_init_fn 控制子进程的随机性
    # 两者结合才能让 DataLoader 的行为完全可复现
    g_loader = torch.Generator()
    g_loader.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
                             worker_init_fn=seed_worker, generator=g_loader)
    valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
                           worker_init_fn=seed_worker) # 验证时不需要反向传播，内存占用较小，可以使用更大的 batch size 来加速评估过程
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
                            worker_init_fn=seed_worker)

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f'train/val/test = {len(trainset)}/{len(valset)}/{len(testset)}')

    net = build_model(num_classes=10).to(device)

    # ===== 修改6: 损失函数升级（Label Smoothing + 类别权重）=====
    # cat 对应 CIFAR-10 的 index=3，与 dog(index=5) 外形相似导致混淆
    # 给 cat 设置 1.8 倍权重，让模型在猫的样本上产生更大梯度，强制学习细粒度特征
    class_weights = torch.ones(10, device=device)
    class_weights[3] = 1.8  # 放大 cat 类别的损失权重
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ===== 修改7: 优化器改为 AdamW =====
    optimizer = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=5e-4) # AdamW 替代 Adam，具有更好的正则化效果，适合迁移学习场景

    # ===== 修改8: 调度器改为余弦退火（CosineAnnealingLR），与 CutMix/Mixup 更搭配 =====
    num_epochs = 35  # 适当增加 epoch，给模型更多学习细粒度特征的时间
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    ) # 余弦退火让学习率平滑下降到极小值，避免突变，配合混合增强更稳定

    best_val_acc = 0.0
    best_path = 'best_resnet50_transfer.pth'
    cutmix_prob = 0.5  # 每个 batch 有 50% 概率使用 CutMix，否则使用 Mixup

    print('开始训练（含 CutMix/Mixup + 验证集监控）...')
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # ===== 修改10: 训练时随机应用 CutMix 或 Mixup =====
            r = np.random.rand()
            if r < cutmix_prob:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            else:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.8)

            optimizer.zero_grad()
            outputs = net(images)
            # 使用混合标签计算损失
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (lam * (preds == targets_a).float().sum().item()
                                + (1 - lam) * (preds == targets_b).float().sum().item())
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(net, valloader, criterion, device)
        scheduler.step()  # 余弦退火按 epoch 步进

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), best_path)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch [{epoch+1}/{num_epochs}] '
            f'train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% '
            f'val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% '
            f'lr={current_lr:.6f}'
        )

    print('\n训练结束，加载最佳验证集模型并在测试集评估...\n')
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=device))

    test_with_required_classes(net, testloader, device, classes)

    # 保留你原先平台占位输出逻辑
    dummy_data = {'id': [0], 'label': [0]}
    dummy_predictions = pd.DataFrame(dummy_data)
    dummy_predictions.to_csv('submission.csv', index=False)
    print('\n---')
    print('已成功生成submission.csv文件以满足平台要求。')
