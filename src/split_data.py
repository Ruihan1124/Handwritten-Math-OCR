import os
import shutil
import random

# Set dataset paths
dataset_path = r"C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data\dataset"
train_path = r"C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data/train"
test_path = r"C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data/test"
split_ratio = 0.8  # 80% for training, 20% for testing

# Ensure target directories exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Iterate over all categories
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    # Skip non-folder items
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)
    random.shuffle(images)  # Shuffle data randomly
    split_idx = int(len(images) * split_ratio)

    # Create category subfolders in train and test directories
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(test_path, category), exist_ok=True)

    # Move files to train and test folders
    for img in images[:split_idx]:
        shutil.move(os.path.join(category_path, img), os.path.join(train_path, category, img))
    for img in images[split_idx:]:
        shutil.move(os.path.join(category_path, img), os.path.join(test_path, category, img))

print("Dataset split completed!")
