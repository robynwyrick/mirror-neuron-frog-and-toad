#!/usr/bin/env python

import time
import os
import random
import collections

class Flies:
    def __init__(self, width):
        self.width = width
        self.flies = [' '] * width

    def place_flies(self):
        for i in range(self.width):
            if random.random() < 0.01:  # Chance to place a fly
                self.flies[i] = '·'

    def update_flies(self):
        new_flies = [' '] * self.width
        for i in range(self.width):
            if self.flies[i] == '·':
                move = random.choice([-1, 0, 1])
                new_position = max(0, min(self.width - 1, i + move))
                if new_flies[new_position] != '·':
                    new_flies[new_position] = '·'
        self.flies = new_flies

    def is_fly_at_position(self, position):
        return 0 <= position < self.width and self.flies[position] == '·'

    def remove_fly_at_position(self, position):
        if 0 <= position < self.width:
            self.flies[position] = ' '

    def get_display_string(self):
        return ''.join(self.flies)
