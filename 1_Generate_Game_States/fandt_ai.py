#!/usr/bin/env python

import time
from fandt import Game as BaseGame
#from lstm_agent import RLAgent


class Game(BaseGame):

    def process_action(self, character, action):
        """Process a specific action for a character."""
        if action == 'f':
            terrain = self.ground.get_terrain(self.frog.position + 1)
            character.hop(terrain)
        elif action == 't':
            terrain = self.ground.get_terrain(self.toad.position + 1)
            character.hop(terrain)
        elif action in ['F', 'T']:
            self.catch_fly(character)
        elif action in ['l', 'k']:
            terrain = self.ground.get_terrain(character.position + 5)
            character.leap(terrain)
        elif action == 'h': # frog is helping toad
            terrain = self.ground.get_terrain(self.toad.position + 5)
            character.help(self.toad, terrain)
        elif action == 'H': # toad is helping frog
            terrain = self.ground.get_terrain(self.frog.position + 5)
            character.help(self.frog, terrain)

        # Call the necessary methods to update the game world
        self.update_world()
        self.handle_character_animations()

        # Ground adjustment logic
        if action in ['f', 't', 'l', 'k', 'h', 'H']:
            #shift_amount = self.ground.adjust_ground((self.frog.position + self.toad.position) // 2)
            shift_amount = self.ground.adjust_ground((self.frog.position + self.toad.position) // 2, self.initial_time_limit, self.time_limit)
            self.frog.position -= shift_amount
            self.toad.position -= shift_amount
            self.frog.position = max(0, min(self.WINDOW_WIDTH - 1, self.frog.position))
            self.toad.position = max(0, min(self.WINDOW_WIDTH - 1, self.toad.position))

    def API_in(self, identity, action):
        """Process an action from the AI model."""
        #print(f"Agent {identity} performs action {action}")

        # `identity` identifies if the player is 'frog' or 'toad' by a 0 or 1.
        # `action` represents a single digit from 0-3.
        action_mapping = [
            [  # Actions for Frog
                'f',  # hop_frog
                'F',  # jump_frog
                'l',  # leap_frog
                'h',  # help_frog
            ],
            [  # Actions for Toad
                't',  # hop_toad
                'T',  # jump_toad
                'k',  # leap_toad
                'H',  # help_toad
            ]
        ]

        # Map the input action to a key press
        print("this is the identity: ", identity)
        print("this is the action: ", action)
        key_press = action_mapping[identity][action]

        # If a valid action is received, update the game state accordingly
        if key_press:
            character = self.frog if identity == 4 else self.toad
            self.process_action(character, key_press)

            
    def API_out(self, identity):
        """Return the current game state from the perspective of a given player."""
        #print(f"Agent {self.identity} chooses action {action}")

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
        
        '''
        symbol_mapping = {
            'f': 0, # Frog hopping
            't': 1, # Toad hopping
            'J': 2, # Jumping
            'L': 3, # Leaping
            'H': 4, # Helping
            'X': 5, # Hurting
            '∏': 6, # Ground
            '^': 7, # Sharp rock
            '·': 8, # Fly
            ' ': 9  # Empty space
        }
        
        symbol_mapping = {
            'f': 0,  # Frog normal
            't': 1,  # Toad normal
            #'F': 2,  # Frog jumping
            #'T': 3,  # Toad jumping
            'H': 4,  # Helping //Frog helping
            'J': 5,  # Jumping //Toad helping
            'L': 6,  # Leaping //Frog hurting
            'X': 7,  # Hurting //Toad hurting
            '∏': 8,  # Ground
            #'_': 9,  # Pit
            '^': 10, # Sharp rock
            '·': 11, # Fly
            ' ': 12  # Empty space
        }
        '''
        
        # Flatten the game state information
        game_state_flat = []
        self.ground_line_str = self.get_ground_line()
        self.players_line_str = self.get_players_line()
        self.flies_line_str = self.flies.get_display_string()
        #print(f"\n\n1 Players Line: {self.players_line_str}\n")

        # Add individual features for terrain and players
        game_state_flat.extend([symbol_mapping[char] for char in self.ground_line_str])
        game_state_flat.extend([symbol_mapping[char] for char in self.players_line_str])
        #print(f"Game state: {game_state_flat}\n")
        game_state_flat.extend([symbol_mapping[char] for char in self.flies_line_str])

        # Add other state information
        game_state_flat.append(symbol_mapping[self.frog.symbol] if identity == 4 else symbol_mapping[self.toad.symbol]) # frog or toad
        game_state_flat.append(self.frog.energy if identity == 4 else self.toad.energy) # energy
        game_state_flat.append(self.frog.score if identity == 4 else self.toad.score) # score
        #game_state_flat.append(self.time_limit) # time remaining
        game_state_flat.append(self.frog.position if identity == 4 else self.toad.position) # frog or toad position

        return game_state_flat

