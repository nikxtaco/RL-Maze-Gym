import sys
from stable_baselines3 import PPO
from maze_env import MazeEnv
import matplotlib.pyplot as plt
import numpy as np

def test(seed=None):
    # Create environment
    env = MazeEnv()
    env.seed(seed)
    maze = env._generate_maze()
    # Find all the empty positions (not walls)
    valid_positions = np.argwhere(maze == 0)
    # Randomly choose one of the valid positions as the target
    target_position = tuple(valid_positions[np.random.choice(len(valid_positions))])

    env = MazeEnv(seed=seed, target_pos=target_position)
    model = PPO.load("target_maze")

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        plt.pause(0.5)  # To slow down the rendering
        plt.draw()

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    test(seed)
