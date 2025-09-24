import math
import os
import assist
import plot_self
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from Resnet34 import resnet34  # 导入自定义的 ResNet34 模型

# # 计算一个batch中预测正确的个数
# 将预测值y_hat与真实值y进行比较，获取预测正确的个数
# accuracy_num返回的结果除以len(y)，则为准确率
def accurate_num(y_hat, y):
    """计算预测正确的数量"""
    # 这里的y_hat的行数为样本数，列数为分类数，即一行表示某个样本的计算结果（经过softmax后每行所有元素的和都为1）
    # y是1维向量，元素的个数对应样本数
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 判断检查y_hat是一个矩阵
        y_hat = y_hat.argmax(axis=1)  # 获取每一行最大元素值的下标，即预测的分类的类别
    cmp = y_hat.type(y.dtype) == y  # 将y_hat转变为y的数据类型，然后作比较，cmp为布尔类型
    return float(cmp.type(y.dtype).sum())


def train(model, train_dataloader, test_dataloader, num_epochs, loss_fun, optimizer, device):
    """在GPU中训练模型"""
    # 定义存储训练损失、训练精度、测试损失、测试精度的列表
    report_epoch = []  # Epoch级的报告,一个epoch占用一行
    report_batch = []  # Batch级的报告，一个batch占用一行，但不是每一个batch都有，仅包含训练损失、训练精度

    # Xavier Uniform模型初始化，在每一层网络保证输入和输出的方差相同
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)  # 应用Xavier Uniform初始化

    model.to(device)  # 将模型放入GPU

    timer, num_batches = assist.Timer(), len(train_dataloader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

    #  执行每个epoch的训练，得到当前epoch的训练损失和训练精度
    for epoch in range(num_epochs):
        # 打印Epoch状态
        print(f"Epoch {epoch + 1}\n-------------------------------")
        print("learning rate:", optimizer.param_groups[0]['lr'])
        # 将模型设置为训练模式
        model.train()  # 将模型设置为训练模式

        # 实例化Accumulator对象，用于存储训练过程中的状态数据
        metric = assist.Accumulator(3)  # 实例化Accumulator对象，累加器列数为3，训练损失总和、训练准确度总和、样本数

        # 执行每一个batch的训练
        for batch, (X, y) in enumerate(train_dataloader):  # 使用dataloader配合for循环，遍历每个batch
            timer.start()  # 开启定时器
            X, y = X.to(device), y.to(device)  # 将X， y移动至GPU
            # 计算预测值
            y_hat = model(X)
            # 计算损失
            loss_value = loss_fun(y_hat, y)
            # 计算梯度并更新预测值
            if isinstance(optimizer, torch.optim.Optimizer):  # 判断检查updater是否为torch.nn.Module类型
                """使用PyTorch内置的优化器和损失函数"""
                # 清除梯度
                optimizer.zero_grad()
                # 反向传播（计算梯度）
                loss_value.mean().backward()  # 计算整个batch中的损失平均值，mean()与交叉熵中reduction='none'有关
                # 更新参数
                optimizer.step()
            else:
                """使用定制的优化器和损失函数"""
                pass

            with torch.no_grad():
                # 将（batch)训练损失总和、（batch)训练准确度总和、（batch)样本数放入累加器
                metric.add(float(loss_value.sum()), accurate_num(y_hat, y), y.numel())

            timer.stop()  # 关闭定时器

            # 计算一个batch训练loss和accuracy
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            num_batches = len(train_dataloader)
            # %取模:返回除法的余数  //取整除:取商的整数部分
            if (batch + 1) % (num_batches // 5) == 0 or batch == num_batches - 1:
                # 存储至Batch级的报告
                report_batch.append([epoch + (batch + 1) / num_batches, train_loss, train_acc])
                # Batch级绘图
                plot_self.report_plot_batch(num_epochs, report_batch, report_epoch)

        # 利用测试集，评估当前模型，返回当前epoch的测试损失和测试精度
        test_loss, test_acc = evaluate_accuracy(test_dataloader, model, device)

        scheduler.step()

        # 打印Epoch级的报告
        print('train_loss:', train_loss, '\ttrain_acc', train_acc)
        print('test_loss:', test_loss, '\ttest_acc', test_acc)
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')

        # # # 存储至Epoch级的报告
        # report_epoch.append([epoch + 1, train_loss, train_acc, test_loss, test_acc])
        # plot_self.report_plot_batch(num_epochs, report_batch, report_epoch)

        # # 每个Epoch将报告存储至相应csv文件中(路径：/save/文件名.csv)
        # save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        # pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
        #              data=report_epoch).to_csv(save_path + "/report_epoch.csv", index=False)
        # pd.DataFrame(columns=["epoch", "train_loss", "train_acc"],
        #              data=report_batch).to_csv(save_path + "/report_batch.csv", index=False)
        # print("report was saved\n")

        # # 保存每个Epoch的模型参数至文件(路径：/save/model_params_epoch下)
        # epoch_params_file_path = os.path.join(save_path, 'model_params_epoch')
        # if not os.path.exists(epoch_params_file_path):
        #     os.makedirs(epoch_params_file_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        # # epoch_params_file = os.path.join(epoch_params_file_path, 'epoch%d_params.pth' % (epoch + 1))
        # epoch_state = {'model_params': model.state_dict(),
        #                'optimizer_params': optimizer.state_dict(),
        #                'epoch': epoch + 1}
        # torch.save(epoch_state, epoch_params_file_path + '/epoch%d_params.pth' % (epoch + 1))
        # print("model_params_epoch was saved\n")

        # Epoch级绘图
        # plot_self.report_plot_epoch(num_epochs, report_epoch)

# # 计算准确率
# 对于任意数据迭代器dataloader可访问的数据集，评估在任意模型上的准确率
# 实际上是计算一个epoch中的准确率
def evaluate_accuracy(test_dataloader, model, device):
    """计算在指定数据集上模型的精度"""
    if isinstance(model, torch.nn.Module):  # 判断检查model是不是torch.nn.Module类型
        model.eval()  # 将模型设置为评估模式

    # 实例化Accumulator对象，用于存储测试过程中的状态数据
    metric = assist.Accumulator(3)  # 实例化Accumulator对象，累加器列数为3，测试损失总和、测试准确度总和、样本数

    # 执行测试
    with torch.no_grad():  # 不计算梯度，只前向传播
        for X, y in test_dataloader:  # 使用dataloader配合for循环,遍历每个batch
            # 将X，y放入GPU
            if isinstance(X, list):  # 如果X是list类型则按元素移至GPU
                X = [x.to(device) for x in X]
            else:  # 如果X是tensor类型则一次全部移动至GPU
                X = X.to(device)
            y = y.to(device)
            # 计算预测值
            y_hat = model(X)
            # 计算损失
            loss_value = loss_fun(y_hat, y)
            # 将（batch)测试损失总和、（batch)测试准确个数、（batch)样本数放入累加器
            metric.add(float(loss_value.sum()), accurate_num(y_hat, y), y.numel())
    # 返回测试损失和测试精度
    return metric[0] / metric[2], metric[1] / metric[2]


# # 定义超参数
learning_rate = 1e-4
batch_size = 128
num_epochs = 300
weight_decay = 1e-3

# # 创建划分好的训练集和测试集
h_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
v_flip = torchvision.transforms.RandomVerticalFlip(p=0.5)
shape_aug = torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.5, 2))
brightness_aug = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)
train_augs = torchvision.transforms.Compose([h_flip, v_flip])  # 图像增广
train_data_trans = transforms.Compose([transforms.Resize(224),
                                       train_augs,
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_data_trans = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_data = ImageFolder(r'C:\Users\15020\Desktop\leaves\train_images',
                         transform=train_data_trans, target_transform=None)
test_data = ImageFolder(r'C:\Users\15020\Desktop\leaves\test_images',
                        transform=test_data_trans, target_transform=None)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 修正保存路径逻辑
save_dir = r'C:\Users\15020\Desktop\leaves'  # 这是目录路径
file_path = os.path.join(save_dir, 'id_code.csv')  # 这是文件路径

# 检查并创建目录（而不是文件）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 将ImageFolder的映射关系存到csv
id_code = pd.DataFrame(list(train_data.class_to_idx.items()),
                       columns=['label', 'id'])
id_code.to_csv(file_path, index=False)

# # 使用 DataLoaders 为训练/测试准备数据，train_dataloader/test_dataloader是一个可迭代对象
print('-------------------------------\n'
      'Load data\n'
      '-------------------------------')
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print('Train_data:')
print('Number of train_dataloader:\t', len(train_dataloader))  # 显示batch的数量
print('Number of train_dataset:\t', len(train_dataloader.dataset))  # 显示训练集样本总数量
print('Test_data:')
print('Number of test_dataloader:\t', len(test_dataloader))  # 显示batch的数量
print('Number of test_dataset:\t', len(test_dataloader.dataset))  # 显示测试集样本总数量
# 每个batch的数据形状
train_X, train_y = next(iter(train_dataloader))
print('Shape:')
print('The shape of train_features in a batch run:\t', train_X.shape)
print('The shape of train_labels in a batch run:\t', train_y.shape, '\n')

# # 测试一下，第一种读取方法，iter(train_dataloader)返回一个迭代器，使用next()访问
# train_X, train_y = next(iter(train_dataloader))
# print('Train_features in a batch run:', train_X, 'shape:', train_X.shape)
# print('Train_labels in a batch run:', train_y, 'shape:', train_y.shape, '\n')
# # 测试一下，第二种读取方法，使用enumerate(dataloader)配合for循环，遍历每个batch
# for train_batch, (train_X, test_y) in enumerate(train_dataloader):
#     print(train_batch)
#     print('Train_features in a batch run:', train_X, 'shape:', train_X.shape)
#     print('Train_labels in a batch run:', train_y, 'shape:', train_y.shape, '\n')
#     break  # 只显示一个batch
# # 测试一下，第三种读取方法，使用dataloader配合for循环，遍历每个batch
# for (train_X, train_y) in train_dataloader:
#     print('Train_features in a batch run:', train_X, 'shape:', train_X.shape)
#     print('Train_labels in a batch run:', train_y, 'shape:', train_y.shape, '\n')
#     break  # 只显示一个batch


# # 检查torch.cuda是否可用，否则继续使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'-------------------------------\n'
      f'Using {device} device\n'
      f'-------------------------------')

# model = torchvision.models.resnet18(num_classes=176)
model = resnet34(num_classes=train_data_size)

# # 定义损失函数，交叉熵损失
# CrossEntropyLoss无需对label进行onehot编码，并且自带了softmax
# reduction='none'、'mean'、'sum' 指定返回的结果
loss_fun = nn.CrossEntropyLoss(reduction='none')

# # 定义优化器
# 实例化SGD实例
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# # 执行训练
print(f'-------------------------------\n'
      f'Train model\n'
      f'-------------------------------')
print('Start time:', assist.time.strftime('%Y-%m-%d %H:%M:%S', assist.time.localtime(assist.time.time())))  # 打印按指定格式排版的时间
train(model, train_dataloader, test_dataloader, num_epochs, loss_fun, optimizer, device)
print('End time:', assist.time.strftime('%Y-%m-%d %H:%M:%S', assist.time.localtime(assist.time.time())))  # 打印按指定格式排版的时间


# # # 保存模型
# print('-------------------------------\n'
#       'Save model\n'
#       '-------------------------------')
# state = {'model_params': model.state_dict(),
#          'optimizer_params': optimizer.state_dict(), 'epoch': num_epochs}
# torch.save(model.state_dict(), 'save/model_params.pth')
# torch.save(model, 'save/model.pth')

# plt.savefig("save/report.png")
# plt.show()  # 若在pycharm的设置中取消勾选了“在工具窗口中显示绘图”，必须取消注释此行代码，图像才不会一闪而过
print("Done!")