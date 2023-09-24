# Crack_Detection
# Wall Crack Detection using Keras and ResNet50

This repository contains code for training a deep learning model to detect cracks in walls using Keras with the ResNet50 architecture. The model is trained on a dataset of wall images and can be used for automated crack detection in real-world scenarios.

## Getting Started

These instructions will help you set up the environment and train the model on your local machine.

### Prerequisites

To run the code, you need the following libraries and tools installed on your system:

- Python 3.x
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/)

You can install the required Python libraries using `pip`:

```bash
pip install keras tensorflow pillow
```

### Dataset

The dataset used for this project should be organized into two folders: `train` and `valid`. The `train` folder contains training images, and the `valid` folder contains validation images. The dataset should be divided into two classes: one for images with cracks and one for images without cracks.

### Training

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/wall-crack-detection.git
cd wall-crack-detection
```

2. Run the provided Python script to train the model:

```bash
python train.py
```

This script uses the ResNet50 architecture with transfer learning to train a model for wall crack detection. It loads the dataset from the `train` and `valid` folders, preprocesses the images, and saves the trained model as `classifier_resnet_model.h5`.

## Model Evaluation

After training the model, you can evaluate its performance using various metrics such as accuracy, precision, recall, and F1-score. Additionally, you can use the model for making predictions on new images to detect cracks in walls.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The ResNet50 architecture is based on the [Keras Applications](https://keras.io/api/applications/resnet/#resnet50-function) library.
- Dataset used for training and validation (not provided in this repository) should be appropriately sourced and credited.

Feel free to customize and extend this code to suit your specific needs for wall crack detection applications.
