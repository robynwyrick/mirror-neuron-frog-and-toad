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

# Plot Validation Loss on the left y-axis with grayscale and distinct linestyle/marker
ax1.set_xlabel("Epochs", fontsize=14)
ax1.set_ylabel("Validation Loss", fontsize=14)
ax1.plot(
    df['epoch'], df['val_loss'],
    label="Validation Loss",
    color='0.2',           # dark gray
    linestyle='--',        # dashed line
    marker='',            # circle markers
    linewidth=1.5
)
ax1.tick_params(axis='y')
ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

# Create a second y-axis for CMNI
ax2 = ax1.twinx()
ax2.set_ylabel("CMNI", fontsize=14)
ax2.plot(
    df['epoch'], df['cmni'],
    label="CMNI",
    color='0.6',           # lighter gray
    linestyle='-.',        # dash-dot line
    marker='',            # square markers
    linewidth=1.5
)
ax2.tick_params(axis='y')

# Title
fig.suptitle("Validation Loss and CMNI Trends Across Epochs", fontsize=16)

# Legends
lines, labels = ax1.get_lines() + ax2.get_lines(), [l.get_label() for l in ax1.get_lines()] + [l.get_label() for l in ax2.get_lines()]
fig.legend(lines, labels, loc='upper right')

# Save and show the plot in high DPI
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("dual_axis_val_loss_cmni_grayscale.png", dpi=600, format='png')
plt.show()

