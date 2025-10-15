#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
file_path = "./cmni_results.csv"
df = pd.read_csv(file_path)

# Extract epochs and validation loss from checkpoint_id
df['epoch'] = df['checkpoint_id'].str.extract(r'epoch(\d+)', expand=False).astype(int)
df['val_loss'] = df['checkpoint_id'].str.extract(r'-valLoss([\d.]+)', expand=False).astype(float)

# Sort by epoch
df = df.sort_values(by='epoch')

# Create the dual-axis plot
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot Validation Loss on the left y-axis
color = 'tab:red'
ax1.set_xlabel("Epochs", fontsize=14)
ax1.set_ylabel("Validation Loss", color=color, fontsize=14)
ax1.plot(df['epoch'], df['val_loss'], label="Validation Loss", linestyle='-', linewidth=1.5, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Create a second y-axis for CMNI
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel("CMNI", color=color, fontsize=14)
ax2.plot(df['epoch'], df['cmni'], label="CMNI", linestyle='-', linewidth=1.5, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Title
fig.suptitle("Validation Loss and CMNI Trends Across Epochs", fontsize=16)

# Save and show the plot
plt.savefig("dual_axis_val_loss_cmni_trends.png", dpi=600, format='png')
plt.show()

