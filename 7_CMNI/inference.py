#!/usr/bin/env python

import os
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

def run_inference(checkpoint_file, test_data_file, output_dir):
    """
    Runs inference on a given checkpoint file and test data file,
    saves the activations for each layer to CSV files in the output directory.
    
    Parameters:
    - checkpoint_file: Path to the model checkpoint file (.h5).
    - test_data_file: Path to the test data CSV file.
    - output_dir: Directory where the activation CSV files will be saved.
    """
    # Load the pre-trained model
    model = load_model(checkpoint_file, compile=False)
    
    # Load the test data
    df = pd.read_csv(test_data_file)
    
    # Assume that the last column is the target and the rest are features.
    X = df.iloc[:, :-1]  # All columns except the last
    y = df.iloc[:, -1]   # Only the last column
    
    # If your target variable is categorical and not already in a one-hot encoded format, convert it
    yc = to_categorical(y, num_classes=4)
    
    # Predict the dataset results
    y_pred = model.predict(X)
    y_pred_classes = y_pred.argmax(axis=-1)  # Convert predictions to class indices if using categorical output
    
    # Compare predictions with actual labels (optional, for evaluation)
    y_true_classes = yc.argmax(axis=-1)
    
    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    print(f'Checkpoint: {checkpoint_file}')
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')
    
    # Identify all layers from which you want to extract activations, excluding the input layer and Dropout layers
    layer_outputs = [layer.output for layer in model.layers if 'dropout' not in layer.name.lower()]
    
    # Create a new model that will return these outputs, given the model input
    activation_model = Model(inputs=model.inputs[0], outputs=layer_outputs)

    
    # Predict the activations for your entire dataset
    activations = activation_model.predict(X)
    
    # Each element in 'activations' corresponds to a layer's activation
    # and is of shape (num_samples, num_neurons_in_layer)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save activations for each layer
    for i, activation_layer in enumerate(activations):
        # Convert the activations of the current layer to a DataFrame
        layer_df = pd.DataFrame(activation_layer, columns=[f'Neuron_{j}' for j in range(activation_layer.shape[1])])
        
        # Save each layer's activations to a CSV file
        layer_filename = os.path.join(output_dir, f'layer_{i+1}_activations.csv')
        layer_df.to_csv(layer_filename, index=False)

    # Optionally, return metrics or activations if needed
    return accuracy

