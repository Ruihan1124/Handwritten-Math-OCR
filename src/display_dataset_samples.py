import os
import cv2
import matplotlib.pyplot as plt


def visualize_category_samples(dataset_path, num_images=5):
    """Allows users to enter a category index to view sample images of the corresponding category"""

    # get all category
    categories = os.listdir(dataset_path)

    # Display all categories and let the user enter an index
    print("Available categories:")
    for idx, category in enumerate(categories):
        print(f"{idx}: {category}")

    # choose category
    while True:
        try:
            num = int(input("Enter category index to visualize: "))
            if 0 <= num < len(categories):
                break
            else:
                print(f"Invalid input! Please enter a number between 0 and {len(categories) - 1}.")
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

    # get category path
    category = categories[num]
    category_path = os.path.join(dataset_path, category)

    # get some image
    image_files = [os.path.join(category_path, img) for img in os.listdir(category_path)[:num_images]]

    # display image
    fig, axes = plt.subplots(1, len(image_files), figsize=(15, 5))
    for ax, img_path in zip(axes, image_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(os.path.basename(img_path))

    plt.show()


# call function
visualize_category_samples(r"C:\Users\LiRu771\PycharmProjects\Handwritten Math OCR\data\dataset_processed\train")
