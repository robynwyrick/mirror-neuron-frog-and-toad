#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from scipy.stats import kurtosis, skew
from activation_utils import (
    load_game_states,
    get_layer_files,
    parse_layer_id,
    create_directory,
    load_activation_data,
    group_activations
)

# At the top of your script, set hatch line width globally:
mpl.rcParams['hatch.linewidth'] = 8.0  # make hatch strokes thicker


# Define output directories for different statistical methods
OUTPUT_DIRS = {
    'mean': './output_images_mean/',
    'sd': './output_images_sd/',
    'variance': './output_images_variance/',
    'kurtosis': './output_images_kurtosis/',
    'skewness': './output_images_skewness/',
    'boxplot': './output_images_boxplot/'  # Added box plot directory
}

#_BLUE = '#1F77B4' # original colors in the dissertation
#_GREEN = '#28CD1E'
#_RED = '#D14D50'

_BLUE = '#739dbe' # new colors for journal
_GREEN = '#8cff74'
_RED = '#d14d50'

# Create directories if they don't exist
for dir_path in OUTPUT_DIRS.values():
    create_directory(dir_path)

def get_y_axis_limits_with_outliers(activation_values):
    """
    Gets global min and max across all groups to set consistent y-axis limits.

    Parameters:
    - activation_values (dict): Dictionary of activation statistics.

    Returns:
    - tuple: (min_limit, max_limit)
    """
    all_values = np.concatenate(list(activation_values.values()))
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    return (min_val - 0.01 * (max_val - min_val), max_val + 0.01 * (max_val - min_val))









def plot_mean_activations(mean_activations, group, layer_id, y_axis_limit, output_dir,
                          color_map=None, hatch_map=None):
    n = len(mean_activations)
    indices = list(range(n))

    # Default grey
    default_color = '0.6'
    plt.figure(figsize=(15, 9))
    ax = plt.gca()

    # 1) base bars
    base_colors = [color_map.get(i, default_color) for i in indices]
    bars = ax.bar(indices, mean_activations, color=base_colors, edgecolor='k')

    # 2) overlay red slash hatch on just the hatch indices
    for i, bar in enumerate(bars):
        if hatch_map and i in hatch_map:
            # ensure the bar’s face is green
            bar.set_facecolor(color_map.get(i, 'r'))
            # overlay hatch by redrawing the same bar with no fill and hatch='/'
            ax.bar([i], [mean_activations[i]],
                   facecolor='none',
                   edgecolor=_RED,
                   hatch='/',
                   linewidth=0)

    # labels & styling
    ax.set_xlabel('Neuron Index', fontsize=16)
    ax.set_ylabel('Mean Activation', fontsize=16)
    ax.set_title(f'Layer {layer_id} Mean Neuron Activations — {group}',
                 fontsize=18, weight='bold')
    ax.set_ylim(y_axis_limit)
    ax.set_xticks(indices)
    ax.set_xticklabels(indices, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'layer_{layer_id}_mean_{group}.png'),
        dpi=600
    )
    plt.close()




    


def plot_boxplot(activations, group, layer_id, y_axis_limit, output_dir):
    """
    Plots boxplots for activations of a specific group and layer with neuron indices starting at 0.
    """
    plt.figure(figsize=(15, 9))
    
    # Create the boxplot without specifying positions
    box = plt.boxplot(activations, showfliers=True)
    
    # Set labels and title
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation Values')
    plt.title(f'Layer {layer_id} Neuron Activations: {group}')
    
    # Set y-axis limits
    plt.ylim(y_axis_limit)
    
    # Determine the number of neurons
    num_neurons = activations.shape[1]  # Assuming activations is (samples, neurons)
    
    # Set x-ticks: default positions start at 1, so labels should start at 0
    plt.xticks(ticks=range(1, num_neurons + 1), labels=range(num_neurons))
    
    # Optional: Rotate x-tick labels if there are many neurons to improve readability
    # plt.xticks(ticks=range(1, num_neurons + 1), labels=range(num_neurons), rotation=90)
    
    # Ensure layout is tight so labels are not cut off
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'layer_{layer_id}_boxplot_{group}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

def old_plot_boxplot(activations, group, layer_id, y_axis_limit, output_dir):
    """
    Plots boxplots for activations of a specific group and layer.
    """
    plt.figure(figsize=(15, 9))
    plt.boxplot(activations, showfliers=True)
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation Values')
    plt.title(f'Layer {layer_id} Neuron Activations: {group}')
    plt.ylim(y_axis_limit)
    plt.savefig(os.path.join(output_dir, f'layer_{layer_id}_boxplot_{group}.png'))
    plt.close()

def plot_std_dev(std_dev_data, group, layer_id, y_axis_limit, output_dir):
    """
    Plots standard deviation of activations for a specific group and layer.
    """
    plt.figure(figsize=(15, 9))
    plt.bar(range(len(std_dev_data)), std_dev_data, color='blue', alpha=0.6, label='Standard Deviation')
    plt.ylim(y_axis_limit)
    plt.savefig(os.path.join(output_dir, f'layer_{layer_id}_std_dev_{group}.png'))
    plt.close()

def plot_variance(variance_data, group, layer_id, y_axis_limit, output_dir):
    """
    Plots variance of activations for a specific group and layer.
    """
    plt.figure(figsize=(15, 9))
    plt.bar(range(len(variance_data)), variance_data, color='green', alpha=0.6, label='Variance')
    plt.ylim(y_axis_limit)
    plt.savefig(os.path.join(output_dir, f'layer_{layer_id}_variance_{group}.png'))
    plt.close()

def plot_kurtosis(kurtosis_data, group, layer_id, y_axis_limit, output_dir):
    """
    Plots kurtosis of activations for a specific group and layer.
    """
    plt.figure(figsize=(15, 9))
    plt.bar(range(len(kurtosis_data)), kurtosis_data, color='purple', alpha=0.6, label='Kurtosis')
    plt.ylim(y_axis_limit)
    plt.savefig(os.path.join(output_dir, f'layer_{layer_id}_kurtosis_{group}.png'))
    plt.close()

def plot_skewness(skewness_data, group, layer_id, y_axis_limit, output_dir):
    """
    Plots skewness of activations for a specific group and layer.
    """
    plt.figure(figsize=(15, 9))
    plt.bar(range(len(skewness_data)), skewness_data, color='orange', alpha=0.6, label='Skewness')
    plt.ylim(y_axis_limit)
    plt.savefig(os.path.join(output_dir, f'layer_{layer_id}_skewness_{group}.png'))
    plt.close()

def remove_nan_inf(values):
    """
    Replaces NaN and Inf values in an array with zeros for safe plotting.
    """
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values

def process_layer(layer_id, game_states, activation_data):
    """
    Processes a single layer's activation data to compute and plot various statistics.
    """
    # Group activations
    grouped_activations = group_activations(game_states, activation_data, handle_empty=True)
    
    # Calculate mean activations
    mean_activations = {group: np.mean(activations, axis=0) for group, activations in grouped_activations.items()}
    y_axis_mean = get_y_axis_limits_with_outliers(mean_activations)




    # Ensure layer_id is an int
    try:
        lid = int(layer_id)
    except ValueError:
        # If parse_layer_id already gave an int, just use it
        lid = layer_id

    # Define per-layer highlight/color/hatch indices
    if lid == 1:
        blue_idxs  = [0,1,2,4,5,6,8,10,14,15,16]
        green_idxs = [3,7,12,13]
        red_idxs   = [9,11]
        hatch_idxs = []  # none for layer 1
    elif lid == 2:
        blue_idxs  = [2,4,5,6,8,9,10,11,12,13,14,15,16]
        green_idxs = [0,1,3]
        red_idxs   = [7]
        hatch_idxs = [1]  # neuron 1 gets a hatch
    else:
        blue_idxs = green_idxs = red_idxs = hatch_idxs = []

    # Build color_map and hatch_map
    color_map = {}
    for i in blue_idxs:  color_map[i] = _BLUE
    for i in green_idxs: color_map[i] = _GREEN
    for i in red_idxs:   color_map[i] = _RED

    print(f"[DEBUG] Layer {lid} color_map: {color_map}")

    # Choose a hatch style, e.g. '///' cross‐hatch
    hatch_map = { i: '///' for i in hatch_idxs }

    for group, activations in grouped_activations.items():
        plot_mean_activations(
            mean_activations[group],
            group,
            layer_id,
            y_axis_mean,
            OUTPUT_DIRS['mean'],
            color_map=color_map,
            hatch_map=hatch_map
        )

    





    # Box plots with outliers
    y_axis_boxplot = get_y_axis_limits_with_outliers(grouped_activations)
    for group, activations in grouped_activations.items():
        plot_boxplot(activations, group, layer_id, y_axis_boxplot, OUTPUT_DIRS['boxplot'])
    
    # Standard deviation and variance
    std_dev_activations = {group: np.std(activations, axis=0) for group, activations in grouped_activations.items()}
    variance_activations = {group: np.var(activations, axis=0) for group, activations in grouped_activations.items()}
    y_axis_std = get_y_axis_limits_with_outliers(std_dev_activations)
    y_axis_var = get_y_axis_limits_with_outliers(variance_activations)
    for group in grouped_activations.keys():
        plot_std_dev(std_dev_activations[group], group, layer_id, y_axis_std, OUTPUT_DIRS['sd'])
        plot_variance(variance_activations[group], group, layer_id, y_axis_var, OUTPUT_DIRS['variance'])
    
    # Kurtosis and Skewness
    kurtosis_activations = {group: kurtosis(activations, axis=0) for group, activations in grouped_activations.items()}
    skewness_activations = {group: skew(activations, axis=0) for group, activations in grouped_activations.items()}
    
    # Remove NaN and Inf before plotting
    kurtosis_activations = {group: remove_nan_inf(kurtosis_activations[group]) for group in grouped_activations.keys()}
    skewness_activations = {group: remove_nan_inf(skewness_activations[group]) for group in grouped_activations.keys()}
    
    y_axis_kurtosis = get_y_axis_limits_with_outliers(kurtosis_activations)
    y_axis_skewness = get_y_axis_limits_with_outliers(skewness_activations)
    
    for group in grouped_activations.keys():
        plot_kurtosis(kurtosis_activations[group], group, layer_id, y_axis_kurtosis, OUTPUT_DIRS['kurtosis'])
        plot_skewness(skewness_activations[group], group, layer_id, y_axis_skewness, OUTPUT_DIRS['skewness'])

    # Print the statistical data for each group
    print(f"\nStatistical data for Layer {layer_id}:")
    for group in grouped_activations.keys():
        print(f"\nGroup: {group}")
        print(f"\n  Mean activations: {mean_activations[group]}")
        print(f"\n  Std dev activations: {std_dev_activations[group]}")
        print(f"\n  Variance activations: {variance_activations[group]}")
        print(f"\n  Kurtosis activations: {kurtosis_activations[group]}")
        print(f"\n  Skewness activations: {skewness_activations[group]}")

# Main processing loop
if __name__ == "__main__":
    # Load game states
    game_states = load_game_states()
    
    # Get layer files
    layer_files = get_layer_files()
    
    # Process each layer
    for layer_file in layer_files:
        layer_id = parse_layer_id(layer_file)
        print(f"\nProcessing Layer {layer_id}")
        
        # Load activation data for the layer using the utility function
        activation_data = load_activation_data(layer_file)
        
        # Process the layer and print statistical data
        process_layer(layer_id, game_states, activation_data)
    
    print("\nAll layers processed and statistical plots generated.")

