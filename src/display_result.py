import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from test_overall import test_dataset
from train_cnn import model

# 选择测试数据集中的某个样本
sample_image, sample_label = test_dataset[0]  # 取第一张图片
sample_image = sample_image.unsqueeze(0)  # 增加 batch 维度

# 让模型进行预测
model.eval()
with torch.no_grad():
    output = model(sample_image)
    predicted_label = torch.argmax(output, dim=1).item()

# 显示图片和预测结果
plt.imshow(sample_image.squeeze(), cmap="gray")
plt.title(f"Predicted: {predicted_label}, Actual: {sample_label}")
plt.show()
