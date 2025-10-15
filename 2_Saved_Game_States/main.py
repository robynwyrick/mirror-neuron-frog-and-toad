#!/usr/bin/env python

import os
import csv
import random
import shutil

# Directory containing the game state files
input_directory = '../2_Saved_Game_States'
output_directory = './'

# Output file for processed data
processed_file = os.path.join(output_directory, 'processed_game_states.csv')

flies = [
    "3,3,0,0,0,0,0,0,0,0,3,0,0,0,0,3,3,0,0,0,0,3,0,0,0,0,3,0,0,0,0,0",
    "3,0,0,3,0,3,0,0,3,0,0,0,0,3,0,0,0,0,3,3,0,0,0,0,3,0,0,0,3,0,0,0",
    "0,3,0,3,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,3,0,0,0,3,0,0,0,3,3,0,0,3",
    "0,3,0,0,3,0,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,0,0,0,0",
    "0,3,0,0,0,3,0,0,0,0,0,0,3,0,3,0,0,0,3,0,0,0,0,0,0,0,0,0,0,3,0,0",
    "0,3,0,0,0,0,3,0,0,0,0,0,3,0,3,0,0,3,0,0,0,0,0,3,3,0,0,0,3,0,0,0",
]

def adjust_energy():
    """Adjusts the player's energy according to the given prevalence."""
    # Define the ranges and their probabilities
    ranges = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    probabilities = [0.50] + [0.25/4]*4 + [0.15/3]*3 + [0.10/3]*3
    
    # Return a randomly chosen energy value based on the defined probabilities
    return random.choices(ranges, probabilities, k=1)[0]

# Use a set to track unique game states
unique_game_states = set()

# Process each game state file and write directly to the processed file
with open(processed_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    
    for filename in sorted(os.listdir(input_directory)):
        if filename.startswith('game_states_') and filename.endswith('.csv'):
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'r') as infile:
                reader = csv.reader(infile)

                
                for row in reader:
                    # Convert row to integers
                    game_state = list(map(int, row))
                    players = game_state[32:64]
                    player_pos = game_state[99]
                    player_sym = game_state[96]

                    # omit all the Toad records
                    if player_sym != 4: continue

                    # enforce that a player over rough ground is shown in distress
                    # this is a failsafe, in the event that the game state snapshot records their action
                    #   instead of their condition
                    for i in range(0, 32):
                        if players[i] != 0 and game_state[i] != 1: game_state[i+32] = 9

                    # Adjust the player's score to 0
                    game_state[98] = 0
                    
                    # Adjust the player's energy
                    # This simplifies the high variability of `energy` since it is mostly low consequence.
                    game_state[97] = adjust_energy()
                    
                    # Replace the flies section with a randomly chosen string from the flies array
                    # Again, this simplifies the high variability of `flies` since it is mostly low consequence.
                    selected_flies = random.choice(flies).split(',')
                    game_state[64:96] = list(map(int, selected_flies))
                    
                    # Convert game state to tuple and check for uniqueness
                    game_state_tuple = tuple(game_state)
                    if game_state_tuple not in unique_game_states:
                        unique_game_states.add(game_state_tuple)
                        writer.writerow(game_state)

print("Processing complete. Processed data saved to", processed_file)

# Zip the processed game states file
zip_file = processed_file.replace('.csv', '.zip')
shutil.make_archive(processed_file.replace('.csv', ''), 'zip', output_directory, 'processed_game_states.csv')

# Remove the individual game state files
'''
for filename in sorted(os.listdir(input_directory)):
    if filename.startswith('game_states_') and filename.endswith('.csv'):
        os.remove(os.path.join(input_directory, filename))
'''

print(f"{processed_file} zipped and saved as {zip_file}")
print("Original game state files removed.")

