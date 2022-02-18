# -*- coding: utf-8 -*—
# Date: 2022/2/18
# Time: 10:46
# Author: 
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from sc.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 加载数据
train = torchvision.datasets.CIFAR10('../datasets', train=True, transform=torchvision.transforms.ToTensor(),
                                     download=True)
test = torchvision.datasets.CIFAR10('../datasets', train=False, transform=torchvision.transforms.ToTensor(),
                                    download=True)
train_len = len(train)
test_len = len(test)
print(train_len)
print(test_len)
train_data = DataLoader(train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
test_data = DataLoader(test, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 2. 准备模型，损失函数和优化器
learning_rate = 1e-2
HQR_model = HQR_CIFAR10()
Loss = nn.CrossEntropyLoss()
optimizar = torch.optim.SGD(HQR_model.parameters(), lr=learning_rate)
# 放入cuda
HQR_model = HQR_model.to(device)
Loss = Loss.to(device)
# 在SummaryWriter上显示
writer = SummaryWriter('../CIFAR10_logs')

# 3. 准备训练
start_time = time.time()
train_step = 0
test_step = 0
for epoch in range(60):
    print("---------开始进行第{}轮训练----------".format(epoch))
    # 开始训练
    HQR_model.train()
    for data in train_data:
        train_imgs, train_targets = data
        # 放入cuda
        train_imgs = train_imgs.to(device)
        train_targets = train_targets.to(device)
        optimizar.zero_grad()
        train_output = HQR_model(train_imgs)
        train_loss = Loss(train_output, train_targets)
        train_loss.backward()
        optimizar.step()
        end_time = time.time()
        # 显示Loss
        if train_step % 100 == 0:
            print('训练第{}轮,第{}回， 损失值Loss为：{}, 所消耗的时间为：{}'.format(epoch, train_step, train_loss.item(), (end_time-start_time)))
            writer.add_scalar('Train', train_loss, train_step)
        train_step = train_step + 1


    # 开始测试
    HQR_model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_data:
            test_imgs, test_targets = data
            # 放入cuda
            test_imgs = test_imgs.to(device)
            test_targets = test_targets.to(device)
            test_output = HQR_model(test_imgs)
            test_loss = Loss(test_output, test_targets)
            total_test_loss = total_test_loss + test_loss

            test_accuracy = (test_output.argmax(1) == test_targets).sum()
            total_test_accuracy = total_test_accuracy + test_accuracy

    print("测试第{}轮, 测试的结果是：{}, 测试的准确率是：{}".format(test_step, total_test_loss, (total_test_accuracy/test_len)))
    writer.add_scalar('Test', total_test_loss.item(), test_step)
    writer.add_scalar('Accuracy', total_test_accuracy.item(), test_step)
    test_step = test_step + 1
    # 保存模型
    torch.save(HQR_model, '../save_model/HQR_model_第{}轮_准确率为{:.3f}.pth'.format(test_step, (total_test_accuracy/test_len)))
    print("模型已保存！")

writer.close()


