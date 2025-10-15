#!/usr/bin/env python

from datetime import datetime
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam

import random
import itertools

# Ensure the checkpoint directory exists at the beginning
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def get_latest_checkpoint(checkpoint_dir):
    """Return the latest checkpoint file and epoch number based on Unix timestamp."""
    files = os.listdir(checkpoint_dir)
    checkpoint_files = [f for f in files if f.startswith('checkpoint-') and f.endswith('.h5')]
    
    if not checkpoint_files:
        return None, 0  # No checkpoints exist, so return epoch 0

    # Get the latest file based on modification time (Unix timestamp)
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    
    # Extract the epoch number from the filename (assuming filename contains `epochXX-`)
    match = re.search(r'epoch(\d+)', latest_checkpoint)
    if match:
        last_epoch = int(match.group(1))  # Get the epoch number from the filename
    else:
        last_epoch = 0  # Default to 0 if the epoch cannot be found

    return os.path.join(checkpoint_dir, latest_checkpoint), last_epoch

def build_pretrain_model(input_shape, output_shape, dropout_rate=0.15, num_layers=4, num_neurons=32, activation='relu', learning_rate=0.001):
    """Build and compile the neural network model."""
    model = Sequential()
    model.add(Dense(num_neurons, activation=activation, input_shape=(input_shape,), kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))

    for _ in range(num_layers - 1):  # -1 because we've already added the input layer
        model.add(Dense(num_neurons, activation=activation, kernel_regularizer=l2(0.01)))
        model.add(Dropout(dropout_rate))

    # Create an instance of the Adam optimizer with the desired learning rate
    optimizer = Adam(learning_rate=learning_rate)

    model.add(Dense(output_shape, activation='linear'))
    model.compile(optimizer=optimizer, loss='mse')

    return model

class DynamicCheckpoint(Callback):
    def __init__(self, checkpoint_dir, hyperparams_str, monitor='val_loss', mode='min', save_best_only=True):
        super(DynamicCheckpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.hyperparams_str = hyperparams_str
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only

        if self.mode == 'min':
            self.best = float('inf')
        elif self.mode == 'max':
            self.best = -float('inf')
        else:
            raise ValueError("Mode must be 'min' or 'max'")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print(f"Monitor '{self.monitor}' is not available. Skipping checkpoint.")
            return

        save = False
        if self.mode == 'min' and current < self.best:
            self.best = current
            save = True
        elif self.mode == 'max' and current > self.best:
            self.best = current
            save = True

        if save:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Update the valLoss precision to 4 decimal places
            filepath = os.path.join(
                self.checkpoint_dir, 
                f"checkpoint-{timestamp}-{self.hyperparams_str}-epoch{epoch+1:02d}-valLoss{current:.4f}.h5"
            )
            self.model.save(filepath)
            print(f"\nCheckpoint saved to: {filepath}")


# Load the dataframe
df = pd.read_csv('../4_Split_Testing_Data/labeled_game_states_train.csv')

# Assume that the last column is the target and the rest are features.
X = df.iloc[:, :-1]  # All columns except the last
y = df.iloc[:, -1]   # Only the last column

# If your target variable is categorical and not already in a one-hot encoded format, convert it
yc = to_categorical(y, num_classes=4)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, yc, test_size=0.2, random_state=12)

# Try to load the latest checkpoint
latest_checkpoint, last_epoch = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Loading model from checkpoint: {latest_checkpoint}, starting from epoch {last_epoch + 1}")
    pretrain_model = load_model(latest_checkpoint)

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

activation = ['relu']
batch = [20]
dropout = [0.12]
epoch = [500]
layer = [2]
neuron = [17]
rate = [0.000005]

# Create a list of all possible combinations
combinations = list(itertools.product(activation, batch, dropout, epoch, layer, neuron, rate))

# Randomly select 1 combination (adjust the number as needed)
selected_combinations = random.sample(combinations, 1)

for combination in selected_combinations:
    a, b, d, e, l, n, r = combination

    output = ""
    output += f"\n\nactivation = {a}\n"
    output += f"batch = {b}\n"
    output += f"dropout = {d}\n"
    output += f"epoch = {e}\n"
    output += f"layer = {l}\n"
    output += f"neurons = {n}\n"
    output += f"rate = {r}\n"
    print(output)
    
    # Create a hyperparameter string for filenames
    hyperparams_str = f"act{a}_bs{b}_dr{d}_ep{e}_nl{l}_nn{n}_lr{float(r):.0e}"
    
    # Define the DynamicCheckpoint callback
    dynamic_checkpoint_callback = DynamicCheckpoint(
        checkpoint_dir=checkpoint_dir, 
        hyperparams_str=hyperparams_str,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # If no checkpoint was loaded, initialize and train a new model
    if not latest_checkpoint:
        pretrain_model = build_pretrain_model(
            input_shape=X_train.shape[1], 
            output_shape=yc.shape[1], 
            dropout_rate=d, 
            num_layers=l,
            num_neurons=n, 
            activation=a, 
            learning_rate=r
        )

    # Add the callbacks to your model fitting
    pretrain_model.fit(
        X_train, y_train, 
        validation_split=0.2, 
        epochs=e, 
        batch_size=b, 
        verbose=1, 
        callbacks=[early_stopping, dynamic_checkpoint_callback],
        initial_epoch=last_epoch  # Start from the next epoch after the checkpoint
    )

    # Save the final model with a new timestamp
    final_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Calculate final validation loss on the test set (or track it from the last epoch)
    final_val_loss = pretrain_model.evaluate(X_test, y_test, verbose=0)[0]  # Get the loss after training

    fn = os.path.join(
        checkpoint_dir, 
        f"pretrain_model-{final_timestamp}-{hyperparams_str}-epoch{last_epoch+1:02d}-valLoss{final_val_loss:.4f}.h5"
    )

    pretrain_model.save(fn)
    print(f"Final model saved to: {fn}")
 
    # Predict the test set results
    y_pred = pretrain_model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)
    y_test_classes = y_test.argmax(axis=-1)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Output the confusion matrix and other details
    output = f"\n\nHyperparameters: {hyperparams_str}\n"
    output += f"Timestamp: {final_timestamp}\n"
    output += "Confusion Matrix:\n"
    output += str(cm) + "\n\n"
    print(output)
    with open('output.txt', 'a') as f:
        f.write(output)

