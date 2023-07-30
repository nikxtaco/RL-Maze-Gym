import pygame
import gym
import maze_env
from stable_baselines3 import PPO

env = gym.make('Maze-v0', size=5, num_walls=4, agent_id=0)
model = PPO.load("ppo_maze_agent0")

obs = env.reset()

window = pygame.display.set_mode((env.size*100, env.size*100))
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(window)
    pygame.time.wait(100)  # Slow down the game for better visualization
