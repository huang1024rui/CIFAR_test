# -*- coding: utf-8 -*—
# Date: 2022/2/25 0025
# Time: 17:29
# Author: 
import os
import warnings

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.config import args
from model.CIFAR10_model import HQR_CIFAR10


def make_dir():
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)



def train():
    # 1.创建文件夹并指定gpu
    make_dir()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # 2.日志文件
    logs_name = str(args.n_iter) + "_" + str(args.lr) + "_"
    print("log_name: ", logs_name)
    f = open(os.path.join(args.log_dir, logs_name + ".txt"), "w")

    # 3.准备数据
    train = torchvision.datasets.CIFAR10("../Dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                         download=True)
    val = torchvision.datasets.CIFAR10("../Dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
    train_len = len(train)
    val_len = len(val)
    print("The train data len is:{}; The val data len is:{}".format(train_len, val_len))
    train_data = DataLoader(train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    val_data = DataLoader(val, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    # 4.准备模型和相关函数, 并放入gpu中
    layer = [3, 32, 32, 64, 64 * 4 * 4, 64, 10]
    HQR = HQR_CIFAR10(2, layer)
    print(HQR)
    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(HQR.parameters(), lr=args.lr)
    HQR = HQR.to(device)
    Loss = Loss.to(device)
    # 在SummaryWriter上显示
    writer = SummaryWriter('../Logs')

    # 5.训练开始
    train_step = 0
    val_step = 0
    for epoch in range(args.n_iter):
        print("-------------Start the {} training-------------".format(epoch))
        # 开始训练
        HQR.train()
        for tra_data in train_data:
            train_img, train_label = tra_data
            train_img = train_img.to(device)
            train_label = train_label.to(device)

            # 优化迭代
            optimizer.zero_grad()
            train_output = HQR(train_img)
            train_loss = Loss(train_output, train_label)
            train_loss.backward()
            optimizer.step()

            # 显示Loss
            train_step = train_step + 1
            if train_step % 100 == 0:
                print("Training {} Epoch, {} round. The Loss is {:.5f}.".format(epoch, train_step, train_loss.item()))
                writer.add_scalar('Train', train_loss.item(), train_step)

        # 验证开始
        HQR.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        with torch.no_grad():
            for data in val_data:
                val_img, val_label = data
                val_img = val_img.to(device)
                val_label = val_label.to(device)

                # 验证与测试
                val_output = HQR(val_img)
                val_loss = Loss(val_output, val_label)
                total_val_loss = val_loss + total_val_loss

                # 准确率的计算
                val_accuracy = (val_output.argmax(1) == val_label).sum()
                total_val_accuracy = val_accuracy + total_val_accuracy

        # 打印出结果
        print("Validation {} Epoch. The reslut is:{} and the accuracy is:{}".format(val_step, total_val_loss.item(),
                                                                                    (total_val_accuracy/val_len)))
        print("Epoch: %d, total_val_loss： %f, total_val_accuracy：%f, train_loss：%f" % (val_step, total_val_loss.item(), (total_val_accuracy/val_len).item(), train_loss.item()), file=f)
        writer.add_scalar('Validation', total_val_loss.item(), val_step)
        writer.add_scalar('Accuracy', total_val_accuracy.item(), val_step)
        val_step = val_step + 1

        # 保存模型
        torch.save(HQR, '../save_model/HQR_model_Epoch{}_Accu{:.5f}.pth'.format(val_step, (total_val_accuracy/val_len)))
    f.close()

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()






