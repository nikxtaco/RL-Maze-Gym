from maze_env import MazeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Set the seed for reproducibility
SEED = 1234

# Train the model
targets = [[0, 4], [4, 4]]
# Individual environment for each agent
train_env = MazeEnv(size=5, agent_targets=targets, seed=SEED)
train_env = DummyVecEnv([lambda: train_env])

model = PPO("MlpPolicy", train_env, verbose=1)
model.set_random_seed(SEED)
model.learn(total_timesteps=10000)
model.save("maze_model")
