import math
import os
import assist
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
import Resnet34  # 导入自定义的 ResNet34 模型

# 数据集路径
train_dataset_path = r'C:\Users\15020\Desktop\leaves\train_images'
test_dataset_path = r'C:\Users\15020\Desktop\leaves\test_images'

# 数据预处理
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 加载数据集
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform)

# length 长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# print(f"类别数: {len(train_dataset.classes)}")176

# 数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 ResNet34 模型
model = Resnet34.resnet34(num_classes=len(train_dataset.classes))  # 根据类别数动态调整输出层
model = model.to(device)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练参数
epochs = 10
writer = SummaryWriter("./logs_leaves")  # TensorBoard 日志
total_train_step = 0
total_test_step = 0

# 训练和测试循环
for epoch in range(epochs):
    print(f"------- 第 {epoch + 1} 轮训练开始 -------")

    # 训练阶段
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device), targets.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练步数：{total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试阶段
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum().item()

    print(f"测试集 Loss: {total_test_loss}")
    print(f"测试集 Accuracy: {total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

   # # 保存模型
   # torch.save(model.state_dict(), f"resnet34_epoch_{epoch + 1}.pth")
   # print(f"模型已保存：resnet34_epoch_{epoch + 1}.pth")

writer.close()

