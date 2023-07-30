import sys
from stable_baselines3 import PPO
from maze_env import MazeEnv
import matplotlib.pyplot as plt

def test(seed=None):
    # Create environment
    env = MazeEnv(seed=seed)
    model = PPO.load("ppo_maze")

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
