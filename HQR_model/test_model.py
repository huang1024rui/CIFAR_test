# -*- coding: utf-8 -*—
# Date: 2022/2/18 0018
# Time: 16:44
# Author: 黄麒睿
import torch
import torchvision
from PIL import Image
# 1. 读取数据和模型
model = torch.load('../save_model/HQR_model_第19轮_准确率为0.811.pth', map_location=torch.device('cpu'))
img = Image.open('../Image/dog.jpg')
print(img)
img = img.convert('RGB')
print(img)

# 2. 数据格式化
transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

img_tensor = transformer(img)
print(img_tensor.shape)

img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))
print(img_tensor.shape)

# 3. 测试数据
output = model(img_tensor)
print(output)
target = output.argmax(1)
print(target)
