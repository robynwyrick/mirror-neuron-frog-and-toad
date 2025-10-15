#!/usr/bin/env python

import numpy as np
import os
from scipy.stats import entropy
from activation_utils import (
    load_game_states,
    get_layer_files,
    parse_layer_id,
    create_directory,
    load_activation_data,
    group_activations,
    define_scenario_pairs,
    plot_statistic,
    save_plot
)

# Define output directories for different statistical methods
OUTPUT_DIRS = {
    'kl_divergence': './output_images_kl_divergence/'
}

# Create directories if they don't exist
for dir_path in OUTPUT_DIRS.values():
    create_directory(dir_path)

def compute_kl_divergence(hist_p, hist_q):
    """
    Computes the Kullback-Leibler divergence between two histograms.

    Parameters:
    - hist_p (np.ndarray): Histogram counts for distribution P.
    - hist_q (np.ndarray): Histogram counts for distribution Q.

    Returns:
    - float: KL divergence value.
    """
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    hist_p = hist_p + epsilon
    hist_q = hist_q + epsilon
    hist_p /= np.sum(hist_p)
    hist_q /= np.sum(hist_q)
    return entropy(hist_p, hist_q)

def compute_common_positive_divergences(kl_divergence_data, scenario_pair_1, scenario_pair_2):
    """
    Computes the common positive divergences between two scenario pairs.
    
    Parameters:
    - kl_divergence_data (dict): Dictionary containing KL divergence lists for all scenario pairs.
    - scenario_pair_1 (tuple): The first scenario pair to compare.
    - scenario_pair_2 (tuple): The second scenario pair to compare.

    Returns:
    - list: A list of common positive divergences across both scenario pairs.
    """
    divergences_1 = kl_divergence_data.get(scenario_pair_1, [])
    divergences_2 = kl_divergence_data.get(scenario_pair_2, [])

    # Ensure that both lists have the same length
    assert len(divergences_1) == len(divergences_2), "Divergence lists must be of the same length."

    common_positive_divergences = []
    for divergence_1, divergence_2 in zip(divergences_1, divergences_2):
        # Check for common positive divergences
        if divergence_1 > 0 and divergence_2 > 0:
            common_positive_divergences.append(min(divergence_1, divergence_2))
        else:
            common_positive_divergences.append(0)

    return common_positive_divergences


def remove_nan_inf(values):
    """
    Replaces NaN and Inf values in a list with zeros for safe processing.
    """
    return [0 if np.isnan(v) or np.isinf(v) else v for v in values]

# Load the game states
game_states = load_game_states()

# Get the layer files
layer_files = get_layer_files()

# Define the scenario pairs
scenario_pairs = define_scenario_pairs()

# Main processing loop
for layer_file in layer_files:
    layer_id = parse_layer_id(layer_file)
    print(f"\nProcessing Layer {layer_id}")

    # Load activation data using the utility function
    activation_data = load_activation_data(layer_file)

    # Group activations with handle_empty=False to match original behavior
    grouped_activations = group_activations(game_states, activation_data, handle_empty=False)

    # Dictionary to store KL Divergence lists for all scenario pairs
    kl_divergence_data = {}

    # Collect all KL Divergence values to determine y-axis limits
    all_kl_values = []

    # First pass: Compute KL Divergence for all scenario pairs and collect data
    for scenario_a, scenario_b in scenario_pairs:
        print(f"\nComparing {scenario_a} vs {scenario_b}")
        activations_a = grouped_activations.get(scenario_a, np.array([]))
        activations_b = grouped_activations.get(scenario_b, np.array([]))

        # Check if there is data in both scenarios
        if activations_a.size == 0 or activations_b.size == 0:
            print(f"Skipping comparison due to lack of data in one of the scenarios.")
            continue

        num_neurons = activation_data.shape[1]
        kl_divergences = []

        # For each neuron, compute KL divergence
        for neuron_idx in range(num_neurons):
            # Get activation values for the neuron in both scenarios
            activations_neuron_a = activations_a[:, neuron_idx]
            activations_neuron_b = activations_b[:, neuron_idx]

            # Create histograms
            bins = 50  # Number of bins can be adjusted
            min_activation = np.min(activation_data)
            max_activation = np.max(activation_data)
            hist_a, bin_edges = np.histogram(
                activations_neuron_a,
                bins=bins,
                range=(min_activation, max_activation),
                density=True
            )
            hist_b, _ = np.histogram(
                activations_neuron_b,
                bins=bin_edges,
                density=True
            )

            # Compute KL divergence
            kl_div = compute_kl_divergence(hist_a, hist_b)

            # Check if the scenario starts from 'distressed.0_none' and only consider positive divergences
            if scenario_a == 'distressed.0_none':
                # Calculate the mean activation values for the neuron in both scenarios
                mean_activation_a = np.mean(activations_neuron_a)
                mean_activation_b = np.mean(activations_neuron_b)

                # Only consider positive divergences (i.e., mean_activation_b > mean_activation_a)
                if mean_activation_b > mean_activation_a:
                    kl_divergences.append(kl_div)
                    all_kl_values.append(kl_div)
                else:
                    # If the divergence is negative or zero, skip this value
                    kl_divergences.append(0)  # Optionally, append zero to keep indexing consistent
            else:
                # For other scenarios, use the KL divergence value directly
                kl_divergences.append(kl_div)
                all_kl_values.append(kl_div)

        # Store KL Divergence list for the scenario pair
        kl_divergence_data[(scenario_a, scenario_b)] = kl_divergences

    # Output the KL Divergence data for the current layer
    print(f"\nKL Divergence data for Layer {layer_id}:")
    for (scenario_a, scenario_b), kl_divergences in kl_divergence_data.items():
        kl_divergences_safe = remove_nan_inf(kl_divergences)  # Clean NaN and Inf values
        print(f"  KL Divergences for {scenario_a} vs {scenario_b}: {kl_divergences_safe}")

    # Determine y-axis limits based on all KL Divergence values for the layer
    if all_kl_values:
        min_kl = min(all_kl_values)
        max_kl = max(all_kl_values)
        y_axis_limit = (min_kl - 0.01 * abs(min_kl), max_kl + 0.01 * abs(max_kl))
    else:
        # Default y-axis limits if no KL Divergence was computed
        y_axis_limit = (0, 1)

    # Second pass: Plot KL Divergence for each scenario pair with uniform y-axis
    for (scenario_a, scenario_b), kl_divergences in kl_divergence_data.items():
        kl_divergences_safe = remove_nan_inf(kl_divergences)  # Clean NaN and Inf values for plotting
        # Plotting
        plot_statistic(
            kl_divergences_safe,
            layer_id,
            scenario_a,
            scenario_b,
            'KL Divergence',
            OUTPUT_DIRS['kl_divergence'],
            ylabel='KL Divergence',
            y_axis_limit=y_axis_limit  # Pass the uniform y-axis limit
        )

        print(f"Saved plot layer_{layer_id}_kl_{scenario_a}_vs_{scenario_b}.png")

    # Define the two scenario pairs you want to compare for common positive divergences
    comparison_scenario_pair_1 = ('distressed.0_none', 'distressed.1_frog')
    comparison_scenario_pair_2 = ('distressed.0_none', 'distressed.2_toad')

    # Compute the common positive divergences for the two scenario pairs
    common_positive_divergences = compute_common_positive_divergences(
        kl_divergence_data,
        comparison_scenario_pair_1,
        comparison_scenario_pair_2
    )

    # Plot the common positive divergences
    common_positive_divergences_safe = remove_nan_inf(common_positive_divergences)  # Clean NaN and Inf
    plot_statistic(
        common_positive_divergences_safe,
        layer_id,
        f"{comparison_scenario_pair_1[1]} & {comparison_scenario_pair_2[1]}",
        "Common Positive Divergences",
        'Common Positive Divergences',
        OUTPUT_DIRS['kl_divergence'],
        ylabel='KL Divergence',
        y_axis_limit=y_axis_limit  # Pass the uniform y-axis limit
    )
    print(f"Saved plot for common positive divergences between {comparison_scenario_pair_1[1]} and {comparison_scenario_pair_2[1]}")

print(f"\nKL Divergence analysis saved to {OUTPUT_DIRS['kl_divergence']}")

