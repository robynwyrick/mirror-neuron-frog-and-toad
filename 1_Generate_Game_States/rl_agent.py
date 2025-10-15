#!/usr/bin/env python

# rl_agent.py
import numpy as np
import random
import collections
from collections import deque

SCORE_IMPROVEMENT_BONUS = 5
MAX_PAST_SCORES_TRACKED = 10000
NO_SCORE_PENALTY_THRESHOLD = 20

# ANSI escape sequences for colors
green = '\033[92m' # Green text
red = '\033[91m'   # Red text
blue = '\033[94m'  # Blue text
reset = '\033[0m'  # Reset to default terminal color

class RLAgent:
    def __init__(self, name, state_size, action_size, gamma=0.95, time_steps=5):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.time_since_last_score = 0
        self.initial_move_reset = 0
        self.gamma = gamma  # Discount factor for future rewards

        self.past_scores = []  # List to keep track of past scores

        self.initial_moves = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3] * 1  # Sequence of initial moves
        self.initial_move_counter = 0

        self.ttl_optimum = 0
        self.num_steps = 0

        self.epsilon = 1.0  # Starting value of epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.last_actions = collections.deque(maxlen=20)  # Keep track of the last 20 actions
        self.repetitive_action_penalty = 0

        self.output_1 = ''
        self.output_2 = ''

    def log_prob_random(self):
        if self.name == 'Frog': sym = int(0)
        else: sym = 1

        
        rand = random.random()  # Generate a random float between 0 and 1
        if sym == 0: rand = min(0, rand/3)
        if rand < 0.5:  # 50% chance
            return 0
        elif rand < 0.75:  # 25% chance (50% - 75%)
            return 1
        elif rand < 0.875:  # 12.5% chance (75% - 87.5%)
            return 2
        else:  # 6.25% chance (87.5% - 100%)
            return 3

    def act(self, identity, state):
        """Choose a random action."""
        return self.log_prob_random()


