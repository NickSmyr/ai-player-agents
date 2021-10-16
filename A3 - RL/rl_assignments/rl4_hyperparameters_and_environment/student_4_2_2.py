#!/usr/bin/env python3
# rewards: [golden_fish, jellyfish_1, jellyfish_2, ... , step]
rewards = [100, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -1]

# Q learning learning rate
alpha = 0.1

# Q learning discount rate
gamma = 0.1

# Epsilon initial
epsilon_initial = 1.0

# Epsilon final
epsilon_final = 0.0

# Annealing timesteps
# 10 episodes * 100 steps / episode
annealing_timesteps = 1000

# threshold
threshold = 1e-1
