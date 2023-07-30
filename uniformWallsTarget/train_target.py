from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from maze_env import MazeEnv
import numpy as np

# Create environment
env = MazeEnv()
seed = 12345  # You can set a specific seed here within 32-bit integer range
env.seed(seed)

# Randomly choose a target position that doesn't coincide with a wall
# Generate the maze to access the generated structure
maze = env._generate_maze()
# Find all the empty positions (not walls)
valid_positions = np.argwhere(maze == 0)
# Randomly choose one of the valid positions as the target
target_position = tuple(valid_positions[np.random.choice(len(valid_positions))])

# Re-create environment with the chosen target position
env = MazeEnv(target_pos=target_position)
env.seed(seed)

# Create vectorized environment
vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed)

# Train model
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=60)
model.save("target_maze")
