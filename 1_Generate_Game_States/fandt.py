#!/usr/bin/env python

import time
import os
import random
import collections

# ── grace-ful keyboard fallback ────────────────────────────────────────
try:
    # only try if an X display is present
    if os.environ.get("DISPLAY"):
        from pynput import keyboard
    else:
        raise ImportError("no DISPLAY")
except ImportError:
    # dummy listener that never emits events
    class _DummyKeyListener:
        def __init__(self, *_, **__): pass
        def start(self): pass
    class keyboard:  # shadow module-like object
        Listener = _DummyKeyListener
# ----------------------------------------------------------------------


from character import Character
from ground import Ground
from flies import Flies

class Game:
    WINDOW_WIDTH = 32
    TIME_LIMIT = 500001

    def __init__(self):
        self.key_pressed = None
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        self.frog_scores = []
        self.toad_scores = []

        self.setup_game()
        self.action_frog = 0
        self.action_toad = 0
        self.output_f1 = ''
        self.output_f2 = ''
        self.output_t1 = ''
        self.output_t2 = ''
        self.initial_time_limit = self.TIME_LIMIT
        self.ground_line_str = ''
        self.players_line_str = ''
        self.flies_line_str = ''
        
    def on_press(self, key):
        try:
            self.key_pressed = key.char
        except AttributeError:
            pass  # Handle special keys here if needed

    def setup_game(self):
        """Set up or reset the game to its initial state."""    
        center_position = self.WINDOW_WIDTH // 2
        self.frog = Character('Frog', 'f', position=center_position - 1, window_width=self.WINDOW_WIDTH)
        self.toad = Character('Toad', 't', position=center_position, window_width=self.WINDOW_WIDTH)
        self.frog.scores = self.frog_scores        
        self.toad.scores = self.toad_scores        
        self.game_over = False
        self.ground = Ground(self.WINDOW_WIDTH)  # Create an instance of Ground
        self.flies = Flies(self.WINDOW_WIDTH)  # Create an instance of Flies
        self.flies.place_flies()  # Initially place some flies
        self.time_limit = self.TIME_LIMIT

    def reset_game(self,f_move_reset, t_move_reset):
        """Reset the game."""
        if self.frog.score == 0 and self.toad.score == 0: return

        self.frog.ttl_optimum = 0
        self.toad.ttl_optimum = 0

        # Store the scores before resetting
        self.frog_scores.append(str(self.frog.score))
        self.toad_scores.append(str(self.toad.score))
        self.setup_game()

    def catch_fly(self, character):
        """Check if the character can catch a fly while jumping."""
        jump_position = character.jump()
        if self.flies.is_fly_at_position(jump_position):  # Check for fly using Flies class method
            character.eat_fly()
            self.flies.remove_fly_at_position(jump_position)  # Remove the fly using Flies class method

    def handle_character_animations(self):
        # Logic for handling character animations
        for character in [self.frog, self.toad]:
            if self.ground.get_terrain(character.position) == '^': character.animate_hurt()
            elif self.ground.get_terrain(character.position) == '_': character.animate_hurt()

    def get_players_line(self):
        """Generate the display string for the players' line."""
        players_line = [' '] * self.WINDOW_WIDTH

        # Check if each character is in a pit; if not, display them on the players' line
        if not self.frog.in_pit and 0 <= self.frog.position < self.WINDOW_WIDTH:
            players_line[self.frog.position] = self.frog.symbol
        if not self.toad.in_pit and 0 <= self.toad.position < self.WINDOW_WIDTH:
            players_line[self.toad.position] = self.toad.symbol

        return ''.join(players_line)

    def get_ground_line(self):
        """Generate the display string for the ground line."""
        ground_line = list(self.ground.get_display_string())  # Convert to list for manipulation
        for character in [self.frog, self.toad]:
            if character.in_pit and 0 <= character.position < self.WINDOW_WIDTH:
                # Character is in a pit, update the ground line to reflect this
                #ground_line[character.position] = character.symbol.upper()
                ground_line[character.position] = character.symbol
        return ''.join(ground_line)


    def display(self):

        # Clear the screen and print the game state
        os.system('clear')
        print('\033[H', end='')
        print("\n" + self.flies_line_str)  # Print flies
        print(self.players_line_str)  # Print characters
        print(self.ground_line_str)  # Print ground with characters in pits

        # Print the remaining time
        print(f"\nTime remaining: {self.time_limit}\n")

        # Calculate the fraction of the total time represented by self.time_limit
        time_fraction = (500000 - self.time_limit) / 500000
        #print(time_fraction)

        # To avoid division by a very small number or zero, ensure time_fraction is greater than 0
        if time_fraction > 0:
            # Calculate the predicted score by dividing the current score by the time fraction
            fpredict = self.frog.score // time_fraction
            tpredict = self.toad.score // time_fraction
        else:
            # Handle the case where time_fraction is 0 or self.time_limit is not set properly
            fpredict = 0  # Or any other default value or error handling
            tpredict = 0

        # Print agent data with historical scores
        print(f"name: {self.frog.name}")
        print(f"symbol: {self.frog.symbol}")
        print(f"energy: {self.frog.energy}")
        print("score: "+str(self.frog.score) + " - predict: " + str(fpredict))
        # Print the current actions
        if self.action_frog is not None :
            print(f"Current Action Frog: {self.action_frog}")
        print(f"In Pit: {self.frog.in_pit}")
        print(f"Output 1: {self.output_f1}")
        print(f"Output 2: {self.output_f2}")

        print(f"Past scores: {self.frog_scores}")
        print()
        
        print(f"name: {self.toad.name}")
        print(f"symbol: {self.toad.symbol}")
        print(f"energy: {self.toad.energy}")
        #print(f"score: {self.toad.score}")
        print("score: "+str(self.toad.score) + " - predict: " + str(tpredict))
        # Print the current actions
        if self.action_toad is not None:
            print(f"Current Action Toad: {self.action_toad}")
        print(f"In Pit: {self.toad.in_pit}")
        print(f"Output 1: {self.output_t1}")
        print(f"Output 2: {self.output_t2}")

        print(f"Past scores: {self.toad_scores}")
        
                
    def update_world(self):
        """Update the world elements like flies, regardless of player movements."""
        self.flies.place_flies()  # Place new flies at each update
        self.flies.update_flies()  # Update flies using Flies class method
        self.display()
