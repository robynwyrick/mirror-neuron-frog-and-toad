#!/usr/bin/env python

import time
import os
import random
import collections

class Character:
    def __init__(self, name, symbol, position, energy=20, window_width=None):
        self.name = name
        self.symbol = symbol
        self.position = position
        self.energy = energy
        self.score = 0
        self.scores = []
        self.is_jumping = False
        self.is_hurt_animation = False
        self.is_eating_animation = False
        self.is_help_animation = False
        self.is_leap_animation = False
        self.is_jump_animation = False
        self.in_pit = False
        self.is_leaping = False
        self.is_jumping = False
        self.is_helping = False
        self.window_width = window_width  # Store the game world width

    # go forward one position, so long as you have energy, and you're not in a pit.
    def hop(self, landing_terrain):
        self.symbol = self.name[0].lower()
        if self.energy <= 0: return
        if self.position + 1 < self.window_width and not self.in_pit:
            self.score += 1 # every single hop gains 2 points (which means 2 for reward)!!
            self.position += 1
            self.evaluate_landing(landing_terrain)

    def jump(self):
        """Make the character jump to catch flies."""
        self.is_jumping = True
        #self.symbol = 'J' if self.is_jump_animation else self.name[0].lower()
        self.is_jump_animation = True
        self.animate_jump()
        return self.position

    # go forward 5 positions, getting out and over a pit, or a rough patch
    def leap(self, landing_terrain):
        if self.energy <= 0: return
        next_position = self.position + 5
        if next_position < self.window_width:
            #self.symbol = 'L' if self.is_leap_animation else self.name[0].lower()
            self.is_leaping = True
            self.is_leap_animation = True
            self.animate_leap()
            self.energy -= 1 # a big leap takes energy
            self.position = next_position
            self.evaluate_landing(landing_terrain)
            return True

    def help(self, other, landing_terrain):
        """Help another character out of a pit."""
        if self.energy <= 0: return

        # Check if the other character's leap would land them within the game bounds
        target_position_after_leap = other.position + 5
        self.is_helping = True
        #self.symbol = 'H' if self.is_help_animation else self.name[0].lower()
        self.is_help_animation = True
        self.animate_help()
        if target_position_after_leap < self.window_width:
            
            # if the `other` needs any energy, you can give them up to 2
            if other.energy < 20:
                other.energy = min(20, other.energy + 2)
                self.energy -= 1  # Helping costs energy
                
            # there is a random chance that helping makes the other leap
            if random.random() <= 0.25: 
                other.leap(landing_terrain)
            
        self.is_help_animation = not self.is_help_animation

    def hurt(self):
        self.energy -= 1    # hurting reduces energy.
        #self.score -= 1     # hurting reduces score and reward.
        self.animate_hurt() # Add this line to animate hurt immediately after losing energy

    def eat_fly(self):
        self.energy = min(20, self.energy + 4)

    def evaluate_landing(self, landing_terrain):
        """Evaluate the character's landing after a leap."""
        self.in_pit = False
        if landing_terrain == '^':
            self.hurt()  # Hurt if landing on a sharp rock
        elif landing_terrain == '_':
            self.fall_into_pit()  # Fall into a pit if leaping into one

    def animate_leap(self):
        """Animate the character when leap."""
        self.symbol = 'L' 

    def animate_jump(self):
        """Animate the character when jump."""
        self.symbol = 'J' 

    def animate_help(self):
        """Animate the character when help."""
        self.symbol = 'H'

    def animate_hurt(self):
        """Animate the character when hurt."""
        self.symbol = 'X'

    def reset_jump(self):
        """Reset jump state after displaying."""
        #self.is_jumping = False
        self.symbol = self.name[0].lower()  # Change symbol to lower case to indicate jumping
        
    def reset_leap(self):
        """Reset the leap state after displaying."""
        #self.is_leaping = False
        self.symbol = self.name[0].lower()  # Change symbol to lower case to indicate jumping

    def reset_help(self):
        """Reset the leap state after displaying."""
        #self.is_helping = False
        self.symbol = self.name[0].lower()  # Change symbol to lower case to indicate jumping

