# -*- coding: utf-8 -*—
# Date: 2022/2/17 0017
# Time: 9:16
# Author:
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sc.CIFAR10_model import HQR_CIFAR10
import time

# 加载至GPU显示
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 1. 加载和准备数据
train = torchvision.datasets.CIFAR10('../datasets', train=True, transform=torchvision.transforms.ToTensor(),
                                     download=True)
test = torchvision.datasets.CIFAR10('../datasets', train=False, transform=torchvision.transforms.ToTensor(),
                                    download=True)
train_len = len(train)
test_len = len(test)
print("训练数据的长度是：{}".format(train_len))
print("测试数据的长度是：{}".format(test_len))

train_data = DataLoader(train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
test_data = DataLoader(test, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


## 2. 加载网络，优化器和损失函数
learning_rate = 1e-2
HQR_model = HQR_CIFAR10()
Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(HQR_model.parameters(), lr=1e-2)
# 加载至GPU进行运算
HQR_model.to(device)
Loss.to(device)


# 加载至tensorboards显示
writer = SummaryWriter('../CIFAR10_logs')

## 3. 训练和测试
# 设置全局参数

start_time = time.time()
train_step = 0
test_step = 0
for epoch in range(50):
    print("------------开始第{}轮训练-----------".format(epoch+1))
    # 开始训练
    HQR_model.train()
    for data in train_data:
        train_imgs, train_target = data
        train_imgs = train_imgs.to(device)
        train_target = train_target.to(device)
        optimizer.zero_grad()
        train_output = HQR_model(train_imgs)
        train_loss = Loss(train_output, train_target)
        train_loss.backward()
        optimizer.step()
        end_time = time.time()
        writer.add_scalar('Train', train_loss.item(), train_step)
        train_step = train_step + 1
        if train_step % 100 == 0:
            print("训练第{}次，Loss为：{}".format(train_step, train_loss.item()))
            print("训练第{}次，消耗的时间为：{}".format(train_step, (end_time - start_time)))

    # 开始测试
    HQR_model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_data:
            test_imgs, test_targets = data
            test_imgs = test_imgs.to(device)
            test_targets = test_targets.to(device)
            test_outputs = HQR_model(test_imgs)
            test_loss = Loss(test_outputs, test_targets)
            total_test_loss = total_test_loss + test_loss
            test_accuracy = (test_outputs.argmax(1) == test_targets).sum()
            total_test_accuracy = total_test_accuracy + test_accuracy

    print("测试的损失Loss为：{}".format(total_test_loss))
    writer.add_scalar('Test', total_test_loss.item(), test_step)
    print("测试的准确率为：{}".format((total_test_accuracy/test_len)))
    writer.add_scalar('ACCURACY', total_test_accuracy.item(), test_step)
    torch.save(HQR_model, '../save_model/HQR_CIFAR10_第{}轮_准确率是{}.pth'.format(test_step, (total_test_accuracy/test_len)))
    test_step = test_step + 1
    print("模型已保存！")

writer.close()
