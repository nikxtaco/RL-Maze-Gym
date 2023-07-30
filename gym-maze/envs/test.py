import gym
import gym_maze
import numpy as np

class MyAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        # This agent simply chooses an action randomly.
        return self.action_space.sample()

def test_agent(agent, env, num_episodes):
    total_reward = 0
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
    return total_reward / num_episodes  # Return average reward

env = gym.make('MazeEnvRandom5x5-v0')  # Create an instance of the new Maze environment
agent = MyAgent(env.action_space)  # Replace this with your agent class

average_reward = test_agent(agent, env, num_episodes=100)
print(f'Average reward: {average_reward}')
