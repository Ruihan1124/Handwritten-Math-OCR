# Handwritten Math OCR

This project is a Handwritten Mathematical Symbol and Number Recognition system using Convolutional Neural Networks (CNNs). It is trained to recognize digits (0-9), mathematical operators (+, -, *, /, =), and variable symbols (x, y, z) from handwritten images.

## Features
- **Handwritten digit & symbol recognition**
- **Custom-trained CNN model**
- **Evaluation metrics: accuracy and error analysis per class**
- **Visualization of model performance (accuracy & error count charts)**
- **Single image prediction support**

## Dataset
The dataset is sourced from Kaggle:
[Handwritten Math Symbols Dataset](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols)

**License:** GPL-2.0

The dataset is structured using an ImageFolder format with subdirectories for each class label:
```
/data
    /train
        /0
        /1
        /2
        ...
    /test
        /0
        /1
        /2
        ...
```

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8) and install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib pillow
```

### Clone the Repository
```bash
git clone https://github.com/your-username/Handwritten-Math-OCR.git
cd Handwritten-Math-OCR
```

## Usage
### 1. Train the Model
Modify `train.py` as needed and run:
```bash
python train.py
```

### 2. Evaluate the Model
Run `test.py` to evaluate the trained model on the test dataset:
```bash
python test.py
```
It prints per-class accuracy and visualizes accuracy/error count.

### 3. Predict a Single Image
To classify a single handwritten math symbol, use:
```python
from test import predict_single_image
predict_single_image(model, "path/to/image.png")
```

## Model Architecture
The model is based on a CNN with multiple convolutional and pooling layers:
- Convolutional Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Softmax Output

## Results
The test set performance is visualized with:
- **Line Chart** for accuracy per class
- **Bar Chart** for error count per class

## Contributing
Feel free to contribute by improving model performance, optimizing code, or adding new features.

## License
This project is licensed under the **GPL-2.0 License** due to its dependency on the [Handwritten Math Symbols Dataset](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols), which is also under GPL-2.0.

## Author
- [Your Name](https://github.com/your-username)

