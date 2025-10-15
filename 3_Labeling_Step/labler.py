#!/usr/bin/env python

import os
import csv
import random
import zipfile

# Map symbols to integers
symbol_mapping = {
    ' ': 0, # Empty space
    '∏': 1, # Ground
    '^': 2, # Sharp rock
    '·': 3, # Fly
    'f': 4, # Frog hopping
    't': 5, # Toad hopping
    'J': 6, # Jumping
    'L': 7, # Leaping
    'H': 8, # Helping
    'X': 9, # Hurting
}

def extract_zip(zip_file, extract_to):
    """Extracts the zip file to the specified directory."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def read_game_state_file(file_path):
    game_state_data = set()  # Use a set to avoid duplicates

    # Read the "processed_game_states.csv" file line by line
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # Skip the header
        for row in csv_reader:
            # Convert to integers
            game_state = [int(i) for i in row] # [0:99]
            #game_state_safe = [int(i) for i in row] # [0:99]
            # ground   = game_state[0:32] (note, python omits 1 from the end of the range, so [0:31] in php 
            # players  = game_state[32:64] 
            # flies    = game_state[64:96] 
            # other    = game_state[96:99]
            #   player = game_state[96]
            #   energy = game_state[97]
            #   score  = game_state[98] (zeroed out)
            #   pos    = game_state[99] 

            energy = game_state[97]

            frog_pos = game_state[99] + 32 # Get frog's position from game_state[99]
            frog_symbol = game_state[frog_pos] #Get frog's symbol 
            game_state[frog_pos] = 0 # temporarily set the new symbol in frog's position to 0

            # Get toad's position and symbol
            toad_symbol = 5 # default
            found = False
            for i in range(32, 64):
                if game_state[i] != 0:  # toad is found here in the player's line 
                    found = True
                    toad_pos = i
                    toad_symbol = game_state[toad_pos]
            if found == False:
                toad_pos = frog_pos # if we don't find toad's position, that's because frog and toad on at the same position.
                toad_symbol = int(frog_symbol)

            # Let's take a moment to work out our activation label (0=distressed_none, 1=distressed_frog, 2=distressed_toad, 3=distressed_both)
            # We can do this here because it is the one place where we systematically know what integers represent both frog and toad.
            activation_label = 0
            if frog_symbol != 9 and toad_symbol != 9: activation_label = 0 # distressed_none
            elif frog_symbol == 9 and toad_symbol != 9: activation_label = 1 # distressed_frog
            elif frog_symbol != 9 and toad_symbol == 9: activation_label = 2 # distressed_toad
            else: activation_label = 3 # distressed_both
            game_state[98] = activation_label

            # restore the current condition to frog to the players line AND the player reference
            # if we zero these out, the frog model is less certain about its own position
            game_state[frog_pos] = frog_symbol 
            game_state[96] = 0 #frog_symbol 
            game_state[99] = 0 # to zero out the positioning value, which makes model rely on game_state board more

            '''
            print(game_state)
            print("Frog's position: ",frog_pos)
            print("Frog's symbol: ",frog_symbol)
            print("Frog's symbol from the game_state line: ",game_state[frog_pos])
            print("Frog's energy: ",energy)

            print("Toad's position: ",toad_pos)
            print("Toad's symbol: ",toad_symbol)
            print("Frog's symbol: ",frog_symbol," Toad's symbol: ",toad_symbol)
            '''

            # Now, let's get the label: 
            if game_state[97] <= 0: 
                label = 1  # if they have no energy, get energy, jump
                
            elif frog_pos == 63:  
                label = 3  # if they have energy, but no more ground, help

            # This is important. If they are nearly at the end of the game board, 
            # that can only be because the other player is stuck.
            # So, if they are nearly at the end of the game spaces, no where to go, AND
            #   if they are not 9 (in distress), but 9 is in the row somewhere (the other player is in distress)
            elif frog_pos >= 58 and frog_symbol != 9 and toad_symbol == 9:  
                label = 3  
                
            else:
                # get the next position on the ground for frog
                nextpos = frog_pos - 32 + 1  # +1 to get the next possible move
                
                if game_state[nextpos] == 1:   # Ground
                    label = 0  # Yes energy, good ground, hop, YAY!
                    
                elif nextpos + 4 < 31: 
                    label = 2  # Not good ground, Energy, bad ground, room to leap, leap
                    
                else: 
                    label = 0  # default, if I missed something, hop.

            # Step 2: Set the last two values to 0
            #values_with_zeroed_last_two = game_state[:-2] + ['0', '0']
                    


            # Convert to integers and add the label '9' at the end
            #int_row = tuple([int(item) for item in game_state + values_with_zeroed_last_two] + [label])
            int_row = tuple([int(item) for item in game_state] + [label])

            #print(int_row)
            #print("\n\n\n\n")
            #if i > 1000: exit()

            game_state_data.add(int_row)

    return list(game_state_data)  # Convert back to list

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sorted(data))  # Sort data before writing

def zip_file(file_path):
    """Zips the specified file."""
    zip_file_path = file_path + '.zip'
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    print(f"File zipped as {zip_file_path}")

def display_game_states(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Convert row to integers, except for the last label
            int_row = [int(item) for item in row[:-1]]
            label = row[-1]

            # Split the row into ground, players, and flies
            ground_line = int_row[:32]
            players_line = int_row[32:64]

            ground_display = ''.join([symbol_mapping[item] for item in ground_line])
            players_display = ''.join([symbol_mapping[item] for item in players_line])

            print(f"- - - - - - - - - - - - - - - - - - \nGame State: {row}")
            print(f"Label: {label}")
            print(f"Energy: {int_row[97]}\n")
            print(f"{players_display}")
            print(f"{ground_display}\n")

# Set the path to the zip file
zip_file_path = '../2_Saved_Game_States/processed_game_states.zip'
csv_file_path = '../2_Saved_Game_States/processed_game_states.csv'

# Extract the zip file
extract_zip(zip_file_path, '../2_Saved_Game_States/')

# Read the game_state states from the extracted CSV file
game_states = read_game_state_file(csv_file_path)

# Save the processed data to a new CSV file in the current directory
output_file = 'labeled_game_states.csv'
save_to_csv(game_states, output_file)

# Zip the labeled game_state states file
zip_file(output_file)

print(f"Processing complete. Labeled data saved to {output_file} and zipped.")

# Uncomment the following line to display the game_state states
#file_path = 'labeled_game_states.csv'
#display_game_states(file_path)

