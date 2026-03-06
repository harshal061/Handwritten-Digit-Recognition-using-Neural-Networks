import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

try:
    # Load the training data from the CSV file
    train_data = pd.read_csv('mnist_train.csv', header=None, dtype=np.float32)
    # Load the test data from the CSV file
    test_data = pd.read_csv('mnist_test.csv', header=None, dtype=np.float32)

    # --- 1. Data Preprocessing ---

    # Separate labels (first column) and pixel values (remaining 784 columns)
    y_train = train_data.iloc[:, 0].values
    X_train = train_data.iloc[:, 1:].values

    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values

    # Normalize pixel values from the [0, 255] range to the [0, 1] range
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode the labels (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0])
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)


    # --- 2. Build the Neural Network Model ---

    model = Sequential([
        # Flatten the 784 pixels (28x28) into a 1D array
        Flatten(input_shape=(784,)),
        
        # First hidden layer with 128 neurons and ReLU activation
        Dense(128, activation='relu'),
        
        # Second hidden layer with 64 neurons and ReLU activation
        Dense(64, activation='relu'),
        
        # Output layer with 10 neurons (one for each digit) and softmax activation
        Dense(num_classes, activation='softmax')
    ])

    # Display a summary of the model's architecture
    model.summary()


    # --- 3. Compile the Model ---

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # --- 4. Train the Model ---

    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1  # Use 10% of the training data for validation
    )
    print("Model training finished.")


    # --- 5. Evaluate the Model ---

    print("\nEvaluating model on the test dataset...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


except FileNotFoundError:
    print("Error: Make sure 'mnist_train.csv' and 'mnist_test.csv' are in the same directory as the script.")
except Exception as e:
    print(f"An error occurred: {e}")

