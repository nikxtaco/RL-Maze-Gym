from maze_env import MazeEnv
from stable_baselines3 import PPO

# Set the seed for reproducibility
SEED = 1234

# Load the models
models = [PPO.load(f"maze_model_{i}") for i in range(2)]

# Testing environment
test_env = MazeEnv(size=5, seed=SEED)

# Reset the environment only once
obs = test_env.reset()
for _ in range(10000):
    actions = [model.predict(obs, deterministic=True)[0] for model in models]
    obs, rewards, dones, info = test_env.step(actions[0], actions[1])
    test_env.render()

test_env.close()
