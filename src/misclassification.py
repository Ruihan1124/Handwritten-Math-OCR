import torch
import matplotlib.pyplot as plt
import numpy as np
from train_cnn import model, device, train_dataset  # 仅导入模型和设备，不导入 train() 函数
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 加载测试数据集（修改为你的测试数据路径）
processed_test_path = "C:/Users/LiRu771/PycharmProjects/Handwritten Math OCR/data/dataset_processed/test"
test_dataset = ImageFolder(root=processed_test_path, transform=train_dataset.transform)  # 保持和训练一致的预处理
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 确保模型处于评估模式
model.eval()

# 记录误分类的图片和标签
misclassified_images = []
misclassified_labels = []
correct_labels = []

# 统计每个类别的误分类数量
class_errors = {cls: 0 for cls in test_dataset.classes}
class_totals = {cls: 0 for cls in test_dataset.classes}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            true_label = test_dataset.classes[labels[i].item()]
            pred_label = test_dataset.classes[predicted[i].item()]
            class_totals[true_label] += 1  # 记录该类别的总数

            if predicted[i] != labels[i]:  # 如果分类错误
                misclassified_images.append(images[i].cpu())
                misclassified_labels.append(pred_label)
                correct_labels.append(true_label)
                class_errors[true_label] += 1  # 记录错误数

# 计算每个类别的正确率
class_accuracies = {cls: (class_totals[cls] - class_errors[cls]) / class_totals[cls] for cls in test_dataset.classes}

# **显示部分误分类的图片**
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified_images):
        img = misclassified_images[i].squeeze(0)  # 移除 batch 维度
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Predicted: {misclassified_labels[i]}\nActual: {correct_labels[i]}")
        ax.axis("off")

plt.show()

# **绘制正确率和错误数量**
fig, ax1 = plt.subplots(figsize=(12, 6))

# 转换为列表，方便绘图
class_labels = list(class_accuracies.keys())
accuracy_values = [class_accuracies[cls] for cls in class_labels]
misclassified_values = [class_errors[cls] for cls in class_labels]

# 柱状图（错误分类数量）
ax1.bar(class_labels, misclassified_values, color='red', alpha=0.6, label='Misclassified Count')

# 折线图（正确率）
ax2 = ax1.twinx()
ax2.plot(class_labels, accuracy_values, color='blue', marker='o', linestyle='-', linewidth=2, label='Accuracy')

# 设置标题和轴标签
ax1.set_xlabel("Class Labels")
ax1.set_ylabel("Misclassified Count", color='red')
ax2.set_ylabel("Accuracy", color='blue')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图表
plt.title("Class-wise Accuracy and Misclassification Count")
plt.xticks(rotation=45)
plt.show()
