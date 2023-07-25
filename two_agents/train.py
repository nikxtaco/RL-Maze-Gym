from maze_env import MazeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Set the seed for reproducibility
SEED = 1234

# Train the models
models = []
for i in range(2):
    # Individual environment for each agent
    train_env = MazeEnv(size=5, target_pos=[0, 4] if i == 1 else None, seed=SEED)
    train_env = DummyVecEnv([lambda: train_env])

    model = PPO("MlpPolicy", train_env, verbose=1)
    model.set_random_seed(SEED)
    model.learn(total_timesteps=100000)
    model.save(f"maze_model_{i}")

    models.append(model)
