# activation_utils.py

import os
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Constants
#INPUT_FILE = '../4_Split_Testing_Data/labeled_game_states_test.csv'
INPUT_FILE = '../4_Split_Testing_Data/labeled_game_states_test.scenario.csv'
ACTIVATION_FOLDER = '../6_Layer_Activations/'
'''
ACTIVATION_FOLDER = 'activations/'
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs20_dr0.12_ep500_nl2_nn10_lr5e-06-20241004-181040-epoch89-valLoss0.0497/'
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs20_dr0.12_ep500_nl2_nn11_lr5e-05-20241005-001151-epoch06-valLoss0.0500'
ACTIVATION_FOLDER = 'activations/checkpoint-20241005-225325-actrelu_bs20_dr0.12_ep500_nl3_nn11_lr5e-05-epoch10-valLoss0.0623'
ACTIVATION_FOLDER = 'activations/checkpoint-20241008-091810-actrelu_bs20_dr0.12_ep500_nl2_nn50_lr4e-05-epoch10-valLoss0.0301'
ACTIVATION_FOLDER = 'activations/checkpoint-20241009-193150-actrelu_bs25_dr0.12_ep500_nl3_nn10_lr5e-06-epoch09-valLoss0.0729'
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs26_dr0.12_ep500_nl3_nn6_lr5e-06-20240924-100238-epoch07-valLoss0.46'
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs26_dr0.12_ep500_nl2_nn12_lr2e-06-20240925-005009-epoch01-valLoss0.90'
ACTIVATION_FOLDER = 'activations/checkpoints-20240922-115653-actrelu_bs26_dr0.12_ep500_nl2_nn9_lr3e-06-epoch150-valLoss0.05-epoch11-valLoss0.0600'
'''

'''
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs20_dr0.12_ep500_nl2_nn9_lr4e-06-20240929-232006-epoch11-valLoss0.06'
ACTIVATION_FOLDER = 'activations/checkpoint-20241008-232412-actrelu_bs25_dr0.12_ep500_nl1_nn16_lr4e-06-epoch11-valLoss0.0497'
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs20_dr0.12_ep500_nl2_nn9_lr6e-06-20241004-104931-epoch24-valLoss0.0452'
ACTIVATION_FOLDER = 'activations/checkpoint-20241009-002756-actrelu_bs25_dr0.12_ep500_nl1_nn15_lr4e-06-epoch12-valLoss0.0499'
'''

'''
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs20_dr0.12_ep500_nl2_nn13_lr5e-05-20241005-024808-epoch07-valLoss0.0471'
#ACTIVATION_FOLDER = 'activations/checkpoint-20241005-223557-actrelu_bs20_dr0.12_ep500_nl3_nn11_lr5e-05-epoch01-valLoss0.0774'
ACTIVATION_FOLDER = 'activations/checkpoint-20241009-155505-actrelu_bs25_dr0.12_ep500_nl1_nn10_lr4e-06-epoch22-valLoss0.0536'
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs20_dr0.12_ep500_nl2_nn10_lr5e-06-20241004-160852-epoch09-valLoss0.0592'
ACTIVATION_FOLDER = 'activations/checkpoint-20241009-120054-actrelu_bs25_dr0.12_ep500_nl1_nn12_lr4e-06-epoch04-valLoss0.0595'
ACTIVATION_FOLDER = 'activations/checkpoint-actrelu_bs20_dr0.12_ep500_nl2_nn9_lr4e-06-20240929-232006-epoch26-valLoss0.05'
ACTIVATION_FOLDER = 'activations/checkpoint-20241005-153646-actrelu_bs20_dr0.12_ep500_nl2_nn14_lr5e-05-epoch03-valLoss0.0496'
# √ ACTIVATION_FOLDER = 'activations/checkpoint-20241009-093028-actrelu_bs25_dr0.12_ep500_nl1_nn13_lr4e-06-epoch05-valLoss0.0599'
ACTIVATION_FOLDER = 'activations/checkpoint-20241010-011508-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch10-valLoss0.0556'
ACTIVATION_FOLDER = 'activations/checkpoint-20241010-075609-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch303-valLoss0.0380'
ACTIVATION_FOLDER = 'activations/checkpoint-20241010-114841-actrelu_bs25_dr0.12_ep500_nl2_nn19_lr4e-06-epoch35-valLoss0.0471'
ACTIVATION_FOLDER = 'activations/checkpoint-20241010-121238-actrelu_bs25_dr0.12_ep500_nl2_nn20_lr4e-06-epoch02-valLoss0.0722'
ACTIVATION_FOLDER = 'activations/checkpoint-20241010-125202-actrelu_bs25_dr0.12_ep500_nl2_nn20_lr4e-06-epoch31-valLoss0.0474'
ACTIVATION_FOLDER = 'activations/checkpoint-20241005-161650-actrelu_bs20_dr0.12_ep500_nl2_nn15_lr5e-05-epoch03-valLoss0.0496'
ACTIVATION_FOLDER = 'activations/checkpoint-20241005-161454-actrelu_bs20_dr0.12_ep500_nl2_nn15_lr5e-05-epoch02-valLoss0.0521'
ACTIVATION_FOLDER = 'activations/checkpoint-20241010-012317-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch16-valLoss0.0517'
'''

ACTIVATION_FOLDER = 'activations/checkpoint-20241010-023625-actrelu_bs25_dr0.12_ep500_nl2_nn17_lr4e-06-epoch70-valLoss0.0440'

def create_directory(path):
    """
    Creates a directory if it doesn't exist.
    
    Parameters:
    - path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_game_states(input_file=INPUT_FILE):
    """
    Loads game state data from a CSV file.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    
    Returns:
    - np.ndarray: Loaded game state data.
    """
    return np.loadtxt(input_file, delimiter=',', skiprows=1)

def get_layer_files(activation_folder=ACTIVATION_FOLDER):
    """
    Retrieves a sorted list of layer activation files.
    
    Parameters:
    - activation_folder (str): Path to the folder containing activation files.
    
    Returns:
    - list: Sorted list of activation filenames.
    """
    return sorted([
        f for f in os.listdir(activation_folder)
        if f.startswith('layer_') and f.endswith('_activations.csv')
    ])

def parse_layer_id(layer_file):
    """
    Extracts the layer ID from an activation filename.
    
    Parameters:
    - layer_file (str): Filename of the activation file.
    
    Returns:
    - str: Extracted layer ID.
    """
    return layer_file.split('_')[1]

def load_activation_data(layer_file, activation_folder=ACTIVATION_FOLDER):
    """
    Loads activation data from a specific layer file.
    
    Parameters:
    - layer_file (str): Filename of the activation file.
    - activation_folder (str): Path to the folder containing activation files.
    
    Returns:
    - np.ndarray: Loaded activation data.
    """
    file_path = os.path.join(activation_folder, layer_file)
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1)
    except Exception as e:
        raise IOError(f"Error loading activation data from {file_path}: {e}")

def group_activations(game_states, activation_data, handle_empty=True):
    """
    Groups activation data based on specific conditions in game states.

    Parameters:
    - game_states (np.ndarray): Array of game state data.
    - activation_data (np.ndarray): Array of activation data corresponding to game states.
    - handle_empty (bool): If True, replace empty groups with zeros to maintain consistent array shapes.

    Returns:
    - dict: A dictionary with grouped activations.
    """
    grouped_activations = {
        'distressed.2_toad': [],  
        'distressed.1_frog': [],
        'distressed.3_both': [], 
        'distressed.0_none': []
    }

    DEBUG_SAMPLES = 0   # change to 0 to silence

    for index, row in enumerate(game_states):
        # Extract relevant slice for players [32:64]
        #print(row)
        scenario = row[98]

        # tiny probe to inspect the first few rows that should have gone to frog / toad
        if DEBUG_SAMPLES:
            print(f"Row {index}: scen={scenario}, slice90_end={row[90:]}")
            DEBUG_SAMPLES -= 1


        if scenario == 0:
            grouped_activations['distressed.0_none'].append(activation_data[index])
        elif scenario == 1:
            #print(row)
            if 5 in row[32:64]: 
                grouped_activations['distressed.1_frog'].append(activation_data[index])
        elif scenario == 2:
            if 4 in row[32:64]: 
                grouped_activations['distressed.2_toad'].append(activation_data[index])
        elif scenario == 3:
            grouped_activations['distressed.3_both'].append(activation_data[index])

    print(len(grouped_activations['distressed.0_none']))
    print(len(grouped_activations['distressed.1_frog']))
    print(len(grouped_activations['distressed.2_toad']))
    print(len(grouped_activations['distressed.3_both']))

    # Convert lists to arrays
    for key in grouped_activations:
        if handle_empty:
            if len(grouped_activations[key]) == 0:
                grouped_activations[key] = np.zeros((1, activation_data.shape[1]))
            else:
                grouped_activations[key] = np.array(grouped_activations[key])
        else:
            grouped_activations[key] = np.array(grouped_activations[key])

    return grouped_activations

def define_scenario_pairs():
    """
    Defines scenario pairs for comparison.
    
    Returns:
    - list of tuples: Each tuple contains two scenario names to compare.
    """
    return [
        ('distressed.0_none', 'distressed.1_frog'),
        ('distressed.0_none', 'distressed.2_toad'),
        ('distressed.0_none', 'distressed.3_both'),
        ('distressed.1_frog', 'distressed.2_toad'),
        # Add more pairs if needed
    ]

def plot_statistic(statistics, layer_id, scenario_a, scenario_b, statistic_name, output_dir, xlabel='Neuron Index', ylabel=None, y_axis_limit=None):
    """
    Plots a given statistical measure across neurons.

    Parameters:
    - statistics (list or np.ndarray): Statistical values per neuron.
    - layer_id (str): Identifier for the layer.
    - scenario_a (str): Name of the first scenario.
    - scenario_b (str): Name of the second scenario.
    - statistic_name (str): Name of the statistic (e.g., 'KL Divergence').
    - output_dir (str): Directory to save the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis. If None, defaults to statistic_name.
    - y_axis_limit (tuple): Tuple of (min, max) for y-axis limits. If None, autoscale.
    """
    if ylabel is None:
        ylabel = statistic_name

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(statistics)), statistics)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Layer {layer_id}: {scenario_a} vs {scenario_b} - {statistic_name}')
    
    if y_axis_limit is not None:
        plt.ylim(y_axis_limit)
    
    plt.tight_layout()
    # Dynamically name the plot file based on statistic
    plot_filename = f'layer_{layer_id}_{statistic_name.lower().replace(" ", "_")}_{scenario_a}_vs_{scenario_b}.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

def save_plot(plot_func, *args, **kwargs):
    """
    A generic wrapper to save plots. This can be expanded for more plot types.

    Parameters:
    - plot_func (callable): The plotting function to execute.
    - *args: Arguments for the plotting function.
    - **kwargs: Keyword arguments for the plotting function.
    """
    plot_func(*args, **kwargs)

