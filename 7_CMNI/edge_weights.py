#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Replace this with the actual path to your model checkpoint (.h5 file)
model_path = '../5_Pretraining/checkpoints/checkpoints-20241010-075609-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch303-valLoss0.0380/checkpoint-20241010-023625-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch70-valLoss0.0440.h5'

# Directory to save the plots
plot_directory = 'edge_weight_plots'
os.makedirs(plot_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Load the pre-trained model
model = load_model(model_path, compile=False)

# Print model summary to verify layers
print(model.summary())

# Identify Dense layers (excluding Dropout layers)
dense_layers = [layer for layer in model.layers if 'dense' in layer.name]

# Verify the indices and layer names
for i, layer in enumerate(dense_layers):
    print(f"Layer {i + 1}: {layer.name}")

# Function to analyze and plot weights between two layers
def analyze_and_plot_weights(from_layer, to_layer, from_layer_idx, to_layer_idx):
    from_weights = from_layer.get_weights()[0]  # Weights matrix
    from_biases = from_layer.get_weights()[1]   # Bias vector

    to_weights = to_layer.get_weights()[0]  # Weights matrix
    to_biases = to_layer.get_weights()[1]   # Bias vector

    # Print shapes to confirm dimensions
    print(f"\nWeights from Layer {from_layer_idx} to Layer {to_layer_idx}:")
    print(f"From Layer {from_layer_idx} weights shape: {from_weights.shape}")  # (prev_layer_neurons, current_layer_neurons)
    print(f"To Layer {to_layer_idx} weights shape: {to_weights.shape}")        # (current_layer_neurons, next_layer_neurons)

    # Analyze the distribution of weights in the 'to_layer'
    flattened_weights = to_weights.flatten()

    # Calculate statistics
    mean_weight = np.mean(flattened_weights)
    std_weight = np.std(flattened_weights)

    print(f"Mean weight in Layer {to_layer_idx}: {mean_weight}")
    print(f"Standard deviation of weights in Layer {to_layer_idx}: {std_weight}")

    # Loop over the specified range of neuron indices for both from_layer and to_layer
    num_from_neurons = from_weights.shape[1]  # Number of neurons in the from_layer
    num_to_neurons = to_weights.shape[1]      # Number of neurons in the to_layer

    for from_neuron_idx in range(num_from_neurons):
        for to_neuron_idx in range(num_to_neurons):
            # Extract the weight connecting the specified neurons
            neuron_to_neuron_weight = to_weights[from_neuron_idx, to_neuron_idx]
            print(f"Weight from Neuron {from_neuron_idx} (Layer {from_layer_idx}) to Neuron {to_neuron_idx} (Layer {to_layer_idx}): {neuron_to_neuron_weight}")

            # Determine how significant the weight is compared to others
            z_score = (neuron_to_neuron_weight - mean_weight) / std_weight
            print(f"Z-score of the weight: {z_score}")

            # Visualize the weight distribution and highlight the specific weight
            plt.figure(figsize=(10, 6))
            plt.hist(flattened_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(neuron_to_neuron_weight, color='r', linestyle='dashed', linewidth=2, label=f'Weight: {neuron_to_neuron_weight:.4f}')
            plt.title(f'Weight Distribution in Layer {to_layer_idx}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.legend()

            # Save the plot with a descriptive filename
            plot_filename = os.path.join(plot_directory, f'Layer{from_layer_idx}_Neuron{from_neuron_idx}_to_Layer{to_layer_idx}_Neuron{to_neuron_idx}.png')
            plt.savefig(plot_filename)
            plt.close()

            print(f"Plot saved as {plot_filename}")

# Analyze connections between Layer 1 and Layer 2
if len(dense_layers) >= 2:
    print("\nAnalyzing connections between Layer 1 and Layer 2:")
    analyze_and_plot_weights(dense_layers[0], dense_layers[1], 1, 2)
else:
    print("Model does not have at least 2 Dense layers.")

# Analyze connections between Layer 2 and Layer 3
if len(dense_layers) >= 3:
    print("\nAnalyzing connections between Layer 2 and Layer 3:")
    analyze_and_plot_weights(dense_layers[1], dense_layers[2], 2, 3)
else:
    print("Model does not have at least 3 Dense layers.")

