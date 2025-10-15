#!/usr/bin/env python

# main.py
import time
from fandt_ai import Game
from rl_agent import RLAgent
import csv


NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 500000
STATE_SIZE = 100
ACTION_SIZE = 4
MAX_TIME_WITHOUT_SCORE = 100000  # Threshold for time without score increase

# Function to write game states to a CSV file
def write_game_states_to_csv(states, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(states)

def train_agent(num_episodes, max_steps_per_episode):
    game = Game()

    time_since_last_score_increase = 0
    state_buffer = []  # Buffer for temporarily storing game states
    buffer_size = 100000  # Number of states to store before writing to CSV
    file_counter = 1  # File naming counter

    # Initialize agents with their respective models
    agent_frog = RLAgent('Frog', state_size=STATE_SIZE, action_size=ACTION_SIZE, gamma=0.95)
    agent_toad = RLAgent('Toad', state_size=STATE_SIZE, action_size=ACTION_SIZE, gamma=0.95)

    # Set checkpoint interval (e.g., save every 100 episodes)
    checkpoint_interval = 1

    for episode in range(num_episodes):
        game.reset_game(agent_frog.initial_move_reset, agent_toad.initial_move_reset)
        total_reward_frog = 0
        total_reward_toad = 0
        time_since_last_score_increase = 0  # Reset counter at the start of each episode

        is_frog_turn = True  # Flag to track turns
        agent_frog.initial_move_counter = 0
        agent_toad.initial_move_counter = 0
        agent_frog.initial_move_reset = 0
        agent_toad.initial_move_reset = 0

        for step in range(max_steps_per_episode):
            if is_frog_turn:
                # Frog's turn
                state_frog = game.API_out(4)
                game.action_frog = agent_frog.act(4, state_frog)
                game.API_in(0, game.action_frog)
                game.toad.reset_jump()
                game.toad.reset_help()
                game.toad.reset_help()
                new_state_frog = game.API_out(4)
                reward_frog = 1
                total_reward_frog += reward_frog

                if new_state_frog[-2] > state_frog[-2]:
                    time_since_last_score_increase = 0  # Reset counter if score increased
                else:
                    time_since_last_score_increase += 1  # Increment counter

                game.output_f1 = agent_frog.output_1
                game.output_f2 = agent_frog.output_2
                
                # Collect current game state
                state_buffer.append(state_frog)

            else:
                # Toad's turn
                state_toad = game.API_out(5)
                game.action_toad = agent_toad.act(5, state_toad)
                game.API_in(1, game.action_toad)
                game.frog.reset_jump()
                game.frog.reset_leap()
                game.frog.reset_leap()
                new_state_toad = game.API_out(5)
                reward_toad = 1
                total_reward_toad += reward_toad

                # Check for score increase
                if new_state_toad[-2] > state_toad[-2]:
                    time_since_last_score_increase = 0  # Reset counter if score increased
                else:
                    time_since_last_score_increase += 1  # Increment counter

                game.output_t1 = agent_toad.output_1
                game.output_t2 = agent_toad.output_2

                # Collect current game state
                state_buffer.append(state_toad)

            # Switch turns
            is_frog_turn = not is_frog_turn

            # Decrement the timer and check for game over condition
            game.time_limit -= 1
            #if game.time_limit <= 0 or game.frog.energy <= 0 or game.toad.energy <= 0:
            if game.time_limit <= 0 :
                game.game_over = True

            if game.game_over:
                break

            # Collect current game state
            state_buffer.append(state_frog)

            # Write to CSV file if buffer is full
            if len(state_buffer) >= buffer_size:
                filename = f'../2_Saved_Game_States/game_states_{file_counter:03d}.csv'
                write_game_states_to_csv(state_buffer, filename)
                state_buffer.clear()  # Clear the buffer for next batch
                file_counter += 1  # Increment file counter

            #time.sleep(0.5)  # Delay; adjust as needed


    # Write any remaining states in the buffer to a CSV file
    if state_buffer:
        filename = f'../2_Saved_Game_States/game_states_{file_counter:03d}.csv'
        write_game_states_to_csv(state_buffer, filename)

train_agent(NUM_EPISODES, MAX_STEPS_PER_EPISODE)

