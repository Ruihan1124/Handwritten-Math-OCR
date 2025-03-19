import os
import cv2
import numpy as np
from tqdm import tqdm

# Set dataset paths (absolute paths to avoid errors)
train_path = r"C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data\train"
test_path = r"C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data\test"
processed_path = r"C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data\dataset_processed"

# Ensure output directory exists
os.makedirs(processed_path, exist_ok=True)


# Preprocessing function
def preprocess_image(image_path, output_path, img_size=(64, 64)):
    if not os.path.exists(image_path):  # Ensure file exists
        print(f"❌ File not found: {image_path}")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"❌ Cannot read image: {image_path}")  # Debug
        return

    img = cv2.resize(img, img_size)  # Resize image
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian Blur to remove noise
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply Otsu's Thresholding

    # Normalize pixel values (convert to 0-1)
    img = img / 255.0

    # Get category name
    category = os.path.basename(os.path.dirname(image_path))

    # Create category folder in output path
    output_category_path = os.path.join(output_path, category)
    os.makedirs(output_category_path, exist_ok=True)

    # Save processed image
    output_file = os.path.join(output_category_path, os.path.basename(image_path))
    cv2.imwrite(output_file, (img * 255).astype(np.uint8))  # Convert back to uint8 and save


# Process all images in train and test sets
for dataset_type in ["train", "test"]:
    input_path = train_path if dataset_type == "train" else test_path  # ✅ Fix: Use correct path
    output_path = os.path.join(processed_path, dataset_type)
    os.makedirs(output_path, exist_ok=True)

    print(f"Processing {dataset_type} images...")
    for category in tqdm(os.listdir(input_path)):
        category_path = os.path.join(input_path, category)
        if not os.path.isdir(category_path):
            continue

        for image_name in os.listdir(category_path):
            if not image_name.lower().endswith((".jpg", ".png")):  # ✅ Ignore non-image files
                print(f"⚠️ Skipping non-image file: {image_name}")
                continue

            image_path = os.path.join(category_path, image_name)
            preprocess_image(image_path, output_path)

print("Preprocessing completed!")