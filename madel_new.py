import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

try:
    # Load the training data from the CSV file with proper dtype, inferring header
    print("Loading training data...")
    train_data = pd.read_csv('mnist_train.csv', dtype=np.float32)

    # Load the test data from the CSV file with proper dtype, inferring header
    print("Loading test data...")
    test_data = pd.read_csv('mnist_test.csv', dtype=np.float32)

    # Drop rows with any NaN values to ensure clean data for model training
    initial_train_rows = len(train_data)
    train_data.dropna(inplace=True)
    if len(train_data) < initial_train_rows:
        print(f"Warning: Dropped {initial_train_rows - len(train_data)} rows with NaN values from training data.")

    initial_test_rows = len(test_data)
    test_data.dropna(inplace=True)
    if len(test_data) < initial_test_rows:
        print(f"Warning: Dropped {initial_test_rows - len(test_data)} rows with NaN values from test data.")

    # --- 1. Data Preprocessing ---

    # Separate labels (first column) and pixel values (remaining 784 columns)
    # Assuming the first column is 'label' and others are pixel data after header removal
    y_train = train_data.iloc[:, 0].values.astype(np.int32)  # Labels should be integers
    X_train = train_data.iloc[:, 1:].values

    y_test = test_data.iloc[:, 0].values.astype(np.int32)
    X_test = test_data.iloc[:, 1:].values

    # Normalize pixel values from the [0, 255] range to the [0, 1] range
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode the labels (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0])
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

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
        validation_split=0.1,  # Use 10% of the training data for validation
        verbose=1
    )
    print("Model training finished.")

    # --- 5. Evaluate the Model ---

    print("\nEvaluating model on the test dataset...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --- 6. Optional: Make a prediction on a single image ---
    print("\nMaking a prediction on the first test image...")
    sample = X_test[0:1]  # Get first image, keep batch dimension
    prediction = model.predict(sample)
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(y_test[0])
    print(f"Predicted digit: {predicted_class}, Actual digit: {actual_class}")

except FileNotFoundError:
    print("Error: Make sure 'mnist_train.csv' and 'mnist_test.csv' are in the same directory as the script.")
    print("You can download MNIST CSV files from: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()  # This will show the full error trace
model.save("digit_model.h5")