# Handwritten Digit Recognition using Neural Networks

A machine learning project that classifies handwritten digits (0–9) using a neural network trained on the MNIST dataset. The model learns to recognize digit patterns from grayscale images and predicts the corresponding digit with high accuracy.

---

## Project Overview

Handwritten digit recognition is a classic problem in computer vision and machine learning. In this project, a multi-layer neural network is trained using the MNIST dataset to classify handwritten digits.

The model learns patterns from pixel values of digit images and predicts the correct label.

This project demonstrates:

- Data preprocessing for image datasets
- Neural network model design
- Training and evaluation of deep learning models
- Model performance analysis

---

## Dataset

The project uses the MNIST dataset, which contains:

- 70,000 grayscale images
- Digits from 0–9
- Image size: 28 × 28 pixels

Dataset split:

- Training samples: 60,000  
- Test samples: 10,000  

Each image represents a handwritten digit and is labeled with the correct digit class.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Model Architecture

The neural network consists of:

- Input layer (flattened 28×28 image pixels)
- Multiple fully connected layers
- ReLU activation function
- Softmax output layer for classification

Optimization details:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

---

## Model Performance

After training the model on the MNIST dataset:

**Test Accuracy: 97.65%**

This indicates the model correctly predicts the digit in most cases.

---

## Project Workflow

1. Load the MNIST dataset
2. Preprocess the image data
3. Normalize pixel values
4. Build the neural network model
5. Train the model on training data
6. Evaluate model performance on test data
7. Predict handwritten digits

---

## Project Structure

```
Handwritten-Digit-Recognition
│
├── notebook.ipynb        # Model training and experimentation
├── train.py              # Neural network training script
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## How to Run the Project

### Clone the repository

```
git clone https://github.com/yourusername/handwritten-digit-recognition.git
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the training script

```
python train.py
```

---

## Learning Outcomes

Through this project, I gained experience in:

- Building neural network models
- Image data preprocessing
- Model training and evaluation
- Working with TensorFlow and Keras

---

## Future Improvements

- Implement Convolutional Neural Networks (CNNs) for higher accuracy
- Build a web interface for digit prediction
- Deploy the model as a web application
