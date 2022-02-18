# -*- coding: utf-8 -*—
# Date: 2022/2/17 0017
# Time: 17:22
# Author:
import torch
import torchvision
from PIL import Image

img_path = "../Image/dog.jpg"
img = Image.open(img_path)
print(img)
img = img.convert('RGB')
print(img)

transfomer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

img_tensor = transfomer(img)
# 加载模型
model = torch.load('../save_model/HQR_CIFAR10_第28轮_准确率是0.6809999942779541.pth', map_location=torch.device('cpu'))
img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    target = model(img_tensor)

print(target)
print(target.argmax(1))
