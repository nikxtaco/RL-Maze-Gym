import gym
import gym_maze
import numpy as np

class MyAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        # This agent simply chooses an action randomly.
        return self.action_space.sample()

    def learn(self, observation, action, reward, next_observation, done):
        # This agent does not learn, so this method does nothing.
        pass

def train_agent(agent, env, num_episodes):
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, next_observation, done)
            observation = next_observation

env = gym.make('MazeEnvRandom5x5-v0')  # Create an instance of the new Maze environment
agent = MyAgent(env.action_space)  # Replace this with your agent class

train_agent(agent, env, num_episodes=1000)
