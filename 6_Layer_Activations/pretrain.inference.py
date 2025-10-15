#!/usr/bin/env python

import os
import re
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


# Function to get the latest checkpoint
def get_latest_checkpoint(checkpoint_dir):
    """Return the latest checkpoint file based on modification time."""
    files = os.listdir(checkpoint_dir)
    checkpoint_files = [f for f in files if f.startswith('checkpoint-') and f.endswith('.h5')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Get the latest file based on modification time
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))

    return os.path.join(checkpoint_dir, latest_checkpoint)

# Define the checkpoint directory
checkpoint_dir = '../5_Pretraining/checkpoints'

# Get the path to the latest checkpoint file
#model_path = get_latest_checkpoint(checkpoint_dir)
model_path = '../5_Pretraining/checkpoints/checkpoints-20241010-075609-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch303-valLoss0.0380/checkpoint-20241010-023625-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch70-valLoss0.0440.h5'



print(f"Loading model from latest checkpoint: {model_path}")

# Load the pre-trained model
model = load_model(model_path,compile=False)

# Load the dataframe
df = pd.read_csv('../4_Split_Testing_Data/labeled_game_states_test.csv')

# Assume that the last column is the target and the rest are features.
X = df.iloc[:, :-1] # All columns except the last
y = df.iloc[:, -1] # Only the last column

# If your target variable is categorical and not already in a one-hot encoded format, convert it
yc = to_categorical(y, num_classes=4)

# X is your Feature matrix of shape (n_samples, n_features)
model.build(input_shape=(None, X.shape[1]))

# … after you do model.build(input_shape=(None, X.shape[1])) …
# force the InputTensor to be created:
_ = model(np.zeros((1, X.shape[1]), dtype=X.values.dtype))

# Predict the dataset results
y_pred = model.predict(X)
y_pred_classes = y_pred.argmax(axis=-1)  # Convert predictions to class indices if using categorical output

# Compare predictions with actual labels (optional, for evaluation)
y_true_classes = yc.argmax(axis=-1)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
cm = confusion_matrix(y_true_classes, y_pred_classes)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{cm}')

# Identify all layers from which you want to extract activations, excluding the input layer and Dropout layers
layer_outputs = [layer.output for layer in model.layers[0:] if 'dropout' not in layer.name.lower()]

# — now that predict() has been called, model.input exists —
# Identify all layers (skip your dropout filter here)
layer_outputs = [ly.output for ly in model.layers if 'dropout' not in ly.name]
# Keras-3: use the first (and only) element of `model.inputs`
activation_model = Model(inputs=model.inputs[0],
                         outputs=[ly.output for ly in model.layers
                                  if 'dropout' not in ly.name])


# Predict the activations for your entire dataset
activations = activation_model.predict(X)

# Each element in 'activations' corresponds to a layer's activation
# and is of shape (num_samples, num_neurons_in_layer)

# If you want to examine or save the activations, you can loop through this list
for i, activation_layer in enumerate(activations):
    # Convert the activations of the current layer to a DataFrame
    layer_df = pd.DataFrame(activation_layer, columns=[f'Neuron_{j}' for j in range(activation_layer.shape[1])])
    
    # Optionally, save each layer's activations to a separate CSV file
    layer_df.to_csv(f'layer_{i+1}_activations.csv', index=False)

