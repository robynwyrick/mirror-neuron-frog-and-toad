#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import glob
from activation_utils import (
    load_game_states,
    get_layer_files,
    parse_layer_id,
    load_activation_data,
    group_activations
)
from inference import run_inference
import re

def extract_hyperparameters(checkpoint_id):
    # Example checkpoint_id format:
    # 'checkpoint-20241005-161257-actrelu_bs20_dr0.12_ep500_nl2_nn15_lr5e-05-epoch01-valLoss0.0555'
    pattern = r'act(\w+)_bs(\d+)_dr([\d.]+)_ep(\d+)_nl(\d+)_nn(\d+)_lr([\de.-]+)'
    match = re.search(pattern, checkpoint_id)
    if match:
        activation, batch_size, dropout, epochs, num_layers, num_neurons, learning_rate = match.groups()
        return {
            'activation': activation,
            'batch_size': int(batch_size),
            'dropout': float(dropout),
            'epochs': int(epochs),
            'num_layers': int(num_layers),
            'num_neurons': int(num_neurons),
            'learning_rate': learning_rate,
        }
    else:
        return {}


# Paths and constants
ACTIVATIONS_DIR = './activations'
#GAME_STATES_FILE = '../4_Split_Testing_Data/labeled_game_states_test.csv'
GAME_STATES_FILE = '../4_Split_Testing_Data/labeled_game_states_test.scenario.csv'
CMNI_RESULTS_FILE = 'cmni_results.csv'
#base_checkpoint_dir = '../5_Pretraining/checkpoints/checkpoints-20241010-075609-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch303-valLoss0.0380'
base_checkpoint_dir = '../5_Pretraining/checkpoints/'

# Ensure output base directory exists
os.makedirs(ACTIVATIONS_DIR, exist_ok=True)

# Load game states
game_states = load_game_states(GAME_STATES_FILE)

# List to store CMNI results
cmni_results = []

# Get all checkpoint directories
checkpoint_dirs = [os.path.join(base_checkpoint_dir, d) for d in os.listdir(base_checkpoint_dir) if d.startswith('checkpoints')]

print(f"Found {len(checkpoint_dirs)} checkpoint directories:")
for d in checkpoint_dirs:
    print("  ", d)


for checkpoint_dir in checkpoint_dirs:
    
    if 'nl2_nn17_lr4e-06-epoch303-valLoss0.038' not in checkpoint_dir :
        continue

    # Get all checkpoint files in the directory
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.h5')))
    print(f"  In {checkpoint_dir}, found {len(checkpoint_files)} .h5 files")

    for checkpoint_file in checkpoint_files:
        # Generate a unique identifier for the checkpoint
        checkpoint_id = os.path.basename(checkpoint_file).replace('.h5', '')
        print(f'Processing checkpoint: {checkpoint_id}')
        
        # Set the output directory for this checkpoint
        output_dir = os.path.join(ACTIVATIONS_DIR, checkpoint_id)
        
        # Check if the activations for this checkpoint already exist to avoid redundant computation
        if not os.path.exists(output_dir):
            # Run inference and save activations
            run_inference(checkpoint_file, GAME_STATES_FILE, output_dir)
        else:
            print(f'Activations for checkpoint {checkpoint_id} already exist. Skipping inference.')
        
        # Now, compute CMNI for this checkpoint
        # Get layer activation files for this checkpoint
        layer_files = get_layer_files(activation_folder=output_dir)
        total_mns = 0
        total_neurons = 0
        for layer_file in layer_files:
            layer_id = parse_layer_id(layer_file)
            activation_data = load_activation_data(layer_file, activation_folder=output_dir)
            # Group activations by scenario
            grouped_activations = group_activations(game_states, activation_data, handle_empty=False)


            # Extract activations for scenarios
            activations_none = np.array(grouped_activations.get('distressed.0_none', []))
            activations_frog = np.array(grouped_activations.get('distressed.1_frog', []))
            activations_toad = np.array(grouped_activations.get('distressed.2_toad', []))
            # Check if activations are available
            if activations_none.size == 0 or activations_frog.size == 0 or activations_toad.size == 0:
                print(f"Skipping layer {layer_id} due to missing activations in one of the scenarios.")
                continue

            print(f"    Layer {layer_id}:")
            print(f"       none: {activations_none.shape}, frog: {activations_frog.shape}, toad: {activations_toad.shape}")

            # Compute mean activations
            mean_none = np.mean(activations_none, axis=0)
            mean_frog = np.mean(activations_frog, axis=0)
            mean_toad = np.mean(activations_toad, axis=0)
            # Compute activation differences
            delta_frog = mean_frog - mean_none
            delta_toad = mean_toad - mean_none
            # Compute MNS for each neuron
            mns = np.minimum(delta_frog, delta_toad)
            # Set MNS to 0 where either delta_frog or delta_toad <= 0
            mns[(delta_frog <= 0) | (delta_toad <= 0)] = 0
            # Aggregate MNS
            total_mns += np.sum(mns)
            total_neurons += len(mns)
        # Compute CMNI for the checkpoint
        cmni = total_mns / total_neurons if total_neurons > 0 else 0
        # Extract hyperparameters from checkpoint_id (optional)
        hyperparams = extract_hyperparameters(checkpoint_id)
        print("     Hyperparams:", hyperparams)

        # Store results
        checkpoint_info = {
            'checkpoint_id': checkpoint_id,
            'cmni': cmni,
            'total_mns': total_mns,
            'total_neurons': total_neurons,
            # Include hyperparameters if extracted
            **hyperparams
        }
        print(f"  → CMNI for {checkpoint_id}: {cmni} (MNS={total_mns}, neurons={total_neurons})")

        print("\n",checkpoint_info,"\ntest1\n")
        cmni_results.append(checkpoint_info)

# After processing all checkpoints, save CMNI results to CSV
df_results = pd.DataFrame(cmni_results)
df_results.to_csv(CMNI_RESULTS_FILE, index=False)
print(f"CMNI results saved to {CMNI_RESULTS_FILE}")

