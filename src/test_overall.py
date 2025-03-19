import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train_cnn import HandwrittenMathCNN


# choose gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocess data
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# load test data
test_dataset = ImageFolder(root="C:/Users/LiRu771/PycharmProjects/Handwritten Math OCR/data/dataset_processed/test",
                           transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# load model
model = HandwrittenMathCNN(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load("C:/Users/LiRu771/PycharmProjects/Handwritten Math OCR/models/model.pth"))
model.to(device)
model.eval()

# calculate accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
