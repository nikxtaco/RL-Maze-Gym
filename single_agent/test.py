from maze_env import MazeEnv
from stable_baselines3 import PPO

# Set the seed for reproducibility
SEED = 1234

# Load the trained model
model = PPO.load("maze_model")

# Testing environment
test_env = MazeEnv(size=5, seed=SEED)

obs = test_env.reset()
for _ in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = test_env.step(action)
    test_env.render()

test_env.close()
