from maze_env import MazeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Set the seed for reproducibility
SEED = 1234

# Training environment
train_env = MazeEnv(size=5, seed=SEED)
train_env = DummyVecEnv([lambda: train_env]) 

model = PPO("MlpPolicy", train_env, verbose=1)

# Set the random seed for the model
model.set_random_seed(SEED)

# Train the model
model.learn(total_timesteps=50000)

# Save the trained model
model.save("maze_model")
