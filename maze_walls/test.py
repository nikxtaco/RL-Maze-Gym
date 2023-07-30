from maze_env import MazeEnv
from stable_baselines3 import PPO
import numpy as np
import pygame

# Set the seed for reproducibility
SEED = 1234

np.random.seed(SEED)

# Load the model
model = PPO.load("maze_model")

# Individual environment for both agents
target_pos = [np.random.randint(5, size=2) for _ in range(2)]
test_env = MazeEnv(size=5, agent_targets=target_pos)

done = False
obs = test_env.reset()

# Render the initial state
test_env.render()

while not done:
    # Model determines the action for both agents
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    test_env.render()

# Add a pause at the end of each run
pygame.time.wait(3000)
