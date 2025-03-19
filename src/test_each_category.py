import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from train_cnn import HandwrittenMathCNN

# Define data transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Ensure size matches training input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize input
])

# Load test dataset
test_dataset = ImageFolder(root="C:/Users/LiRu771/PycharmProjects/Handwritten Math OCR/data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = HandwrittenMathCNN(num_classes=len(test_dataset.classes))  # Number of classes from training
model.load_state_dict(
    torch.load("C:/Users/LiRu771/PycharmProjects/Handwritten Math OCR/models/model.pth", map_location=device))
model.to(device)
model.eval()

# Initialize accuracy and error tracking
class_correct = np.zeros(len(test_dataset.classes))
class_total = np.zeros(len(test_dataset.classes))
class_errors = np.zeros(len(test_dataset.classes))

# Evaluate model performance
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            label = labels[i].item()
            if predicted[i] == labels[i]:
                class_correct[label] += 1
            else:
                class_errors[label] += 1
            class_total[label] += 1

# Print accuracy and error counts per class
accuracies = []
error_counts = []
class_names = test_dataset.classes

for i, class_name in enumerate(class_names):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        accuracies.append(accuracy)
        error_counts.append(class_errors[i])
        print(f"Class {class_name}: Accuracy = {accuracy:.2f}%, Errors = {class_errors[i]}")
    else:
        accuracies.append(0)
        error_counts.append(0)

# Plot accuracy (line chart) and error count (bar chart) in a single figure
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Class")
ax1.set_ylabel("Accuracy (%)", color='b')
ax1.plot(class_names, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_xticks(range(len(class_names)))  # Ensure tick positions match the number of class names
ax1.set_xticklabels(class_names, rotation=45)  # Rotate labels for better visibility
ax1.grid()

ax2 = ax1.twinx()
ax2.set_ylabel("Error Count", color='r')
ax2.bar(class_names, error_counts, alpha=0.6, color='r', label='Error Count')
ax2.tick_params(axis='y', labelcolor='r')

fig.suptitle("Model Performance: Accuracy and Error Count per Class")
fig.tight_layout()
plt.show()


# Function to predict a single image
def predict_single_image(model, image_path):
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    # Retrieve class name
    predicted_label = test_dataset.classes[predicted_class.item()]
    print(f"Predicted class: {predicted_label}")
