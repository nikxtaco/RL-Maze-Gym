from maze_env import MazeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Set the seed for reproducibility
SEED = 1234

# Train the model for agent 2
# Individual environment for the agent
train_env = MazeEnv(size=5, target_pos=[3, 2], num_walls=3, seed=SEED)
train_env = DummyVecEnv([lambda: train_env])

model = PPO("MlpPolicy", train_env, verbose=1)
model.set_random_seed(SEED)
model.learn(total_timesteps=30000)
model.save("ppo_maze_agent1")
