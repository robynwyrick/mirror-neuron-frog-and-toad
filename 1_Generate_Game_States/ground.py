#!/usr/bin/env python

import time
import os
import random
import collections

class Ground:
    def __init__(self, width):
        self.width = width
        self.ground = collections.deque(['∏'] * width)
        self.upcoming_obstacles = collections.deque()
        self.random_rate = 0;

    def add_new_ground_element(self):
        if self.upcoming_obstacles:
            return self.upcoming_obstacles.popleft()
        #elif random.random() < 0.0175:
        elif random.random() < self.random_rate:
            obstacle = random.choice(['^', '^'])
            obstacle_width = random.randint(1, 4)
            for _ in range(obstacle_width):
                self.upcoming_obstacles.append(obstacle)
            self.upcoming_obstacles.append('∏')
            return self.upcoming_obstacles.popleft()
        else:
            return '∏'

    def adjust_ground(self, average_position, time_limit, current_time):
        myratio = (time_limit - current_time)/time_limit
        #self.random_rate = 0.075 + myratio * 0.05
        self.random_rate = min(0.075, self.random_rate + 0.005)
        shift_amount = average_position - self.width // 2
        if shift_amount > 0:
            for _ in range(shift_amount):
                self.ground.popleft()
                new_element = self.add_new_ground_element()
                self.ground.append(new_element)
        elif shift_amount < 0:
            for _ in range(-shift_amount):
                self.ground.pop()
                self.ground.appendleft('∏')
        return shift_amount

    def get_terrain(self, position):
        if 0 <= position < self.width:
            return self.ground[position]
        return None  # or some default value for out-of-bounds

    def get_display_string(self):
        return ''.join(self.ground)

