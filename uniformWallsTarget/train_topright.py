from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from maze_env import MazeEnv

# Create environment
seed = 12345  # You can set a specific seed here within 32-bit integer range
env = MazeEnv(seed=seed, target_type='topright')

# Create vectorized environment
vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed)

# Train model
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("topright_maze")
