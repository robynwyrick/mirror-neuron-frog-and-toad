#!/usr/bin/env python

from tabulate import tabulate
import pandas as pd
import re

# Define the path to your CSV file
CSV_FILE_PATH = 'cmni_results.csv'

# Read the CSV file, handling the header row
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file '{CSV_FILE_PATH}' is empty.")
    exit(1)
except pd.errors.ParserError:
    print(f"Error: The file '{CSV_FILE_PATH}' does not appear to be in CSV format.")
    exit(1)

# Function to parse the checkpoint_id and extract required components
def parse_checkpoint_id(checkpoint_id):
    """
    Parses the checkpoint_id string to extract Layers, Neurons, Epochs, Val Loss, Learning Rate, and Date.

    Parameters:
        checkpoint_id (str): The checkpoint filename string.

    Returns:
        dict: A dictionary with keys 'Layers', 'Neurons', 'Epochs', 'Val Loss', 'Learning Rate', 'Date'.
              Returns None for values that cannot be parsed.
    """
    # Initialize default values
    layers = neurons = epochs = val_loss = learning_rate = date = None

    try:
        # Extract Layers (nl followed by digits)
        layers_match = re.search(r'nl(\d+)', checkpoint_id)
        if layers_match:
            layers = int(layers_match.group(1))

        # Extract Neurons (nn followed by digits)
        neurons_match = re.search(r'nn(\d+)', checkpoint_id)
        if neurons_match:
            neurons = int(neurons_match.group(1))

        # Extract Learning Rate (lr followed by digits, possibly with decimal and exponent)
        lr_match = re.search(r'lr([0-9.eE\-]+)', checkpoint_id)
        if lr_match:
            lr_str = str(lr_match.group(1))
            # Limit to the first 5 characters
            learning_rate = lr_str[:5]

        # Extract all epoch and valLoss pairs
        # Pattern: epoch followed by digits, then valLoss followed by float
        epoch_val_loss_matches = re.findall(r'epoch(\d+)-valLoss([0-9.]+)', checkpoint_id)
        
        if epoch_val_loss_matches:
            # Take the last occurrence
            last_epoch, last_val_loss = epoch_val_loss_matches[-1]
            epochs = int(last_epoch)
            val_loss = float(last_val_loss)
        else:
            # Attempt to extract 'epoch' followed by digits and 'valLoss' separately
            epoch_matches = re.findall(r'epoch(\d+)', checkpoint_id)
            val_loss_matches = re.findall(r'valLoss([0-9.]+)', checkpoint_id)
            if epoch_matches and val_loss_matches:
                epochs = int(epoch_matches[-1])
                val_loss = float(val_loss_matches[-1])

        # Extract Date (first 8-digit number in the format YYYYMMDD)
        date_matches = re.findall(r'(\d{8})', checkpoint_id)
        if date_matches:
            date = date_matches[0]  # Assuming the first 8-digit number is the date

    except Exception as e:
        print(f"Error parsing '{checkpoint_id}': {e}")

    return {
        'Layers': layers,
        'Neurons': neurons,
        'Epochs': epochs,
        'Val Loss': val_loss,
        'Learning Rate': learning_rate,
        'Date': date
    }

# Apply the parsing function to each checkpoint_id
parsed_data = df['checkpoint_id'].apply(parse_checkpoint_id)

# Convert the list of dictionaries into a DataFrame
parsed_df = pd.DataFrame(parsed_data.tolist())

# Concatenate the parsed data with the original DataFrame
df = pd.concat([df, parsed_df], axis=1)

# Rename columns for clarity
df.rename(columns={
    'total_mns': 'MNS',
    'cmni': 'CMNI'
}, inplace=True)

# Convert necessary columns to numeric types
numeric_columns = ['Val Loss', 'CMNI', 'MNS', 'Layers', 'Neurons', 'Epochs']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Optional: Convert 'Date' from 'YYYYMMDD' to 'YYYY-MM-DD' format for readability
def format_date(yyyymmdd):
    if pd.isna(yyyymmdd):
        return None
    try:
        return pd.to_datetime(str(int(yyyymmdd)), format='%Y%m%d').strftime('%Y-%m-%d')
    except ValueError:
        return yyyymmdd  # Return as is if parsing fails

df['Date'] = df['Date'].apply(format_date)

# Filter the DataFrame based on Val Loss and CMNI criteria
filtered_df = df[
    (df['Val Loss'] < 0.06) &
    (df['CMNI'] >= 0.005)
]

# Select the desired columns, including 'checkpoint_id', 'Date', and 'Learning Rate'
final_df = filtered_df[['checkpoint_id', 'Date', 'Learning Rate', 'Layers', 'Neurons', 'Epochs', 'Val Loss', 'MNS', 'CMNI']]

# Group by (Date, Learning Rate, Layers, Neurons) and capture the best CMNI
group_columns = ['Date', 'Learning Rate', 'Layers', 'Neurons']
# Using idxmax to find the index of the maximum CMNI within each group
grouped_indices = final_df.groupby(group_columns)['CMNI'].idxmax()
# Selecting the rows with the best CMNI per group
grouped_df = final_df.loc[grouped_indices].reset_index(drop=True)
    
# Sort the grouped DataFrame by CMNI in descending order
grouped_df_sorted = grouped_df.sort_values(by='CMNI', ascending=False).reset_index(drop=True)
#grouped_df_sorted = grouped_df.sort_values(by=['Layers', 'Neurons'], ascending=[True, False]).reset_index(drop=True)

# Check if the final grouped DataFrame is empty
if grouped_df_sorted.empty:
    print("No checkpoints found with Val Loss < 0.05 and CMNI >= 0.005.")
else:
    # Display the resulting grouped table using tabulate for better readability
    print(tabulate(grouped_df_sorted, headers='keys', tablefmt='psql', showindex=False))
    # Alternatively, without tabulate:
    # print(grouped_df_sorted.to_string(index=False))
    
    # Optionally, export the grouped DataFrame to a CSV file
    # grouped_df_sorted.to_csv('grouped_cmni_results.csv', index=False)

