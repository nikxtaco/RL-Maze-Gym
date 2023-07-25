import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self, size=5, seed=None):
        super(MazeEnv, self).__init__()

        self.size = size # Size of the maze
        self.seed_value = seed # Setting the seed for random number generation
        self.np_random, _ = gym.utils.seeding.np_random(self.seed_value)

        # Define action and observation space
        # The action space is a discrete space of size 4 (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # The observation space is a 3-channel image (RGB) of the same size as the maze
        self.observation_space = spaces.Box(low=0, high=3, shape=(size, size, 3), dtype=np.uint8) 

        # Initialize the state of the environment
        self.reset()
        
        self.viewer = None

    def step(self, action):
        # Update the agent's position based on the action taken
        if action == 0: # Up
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        elif action == 1: # Down
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0]+1)
        elif action == 2: # Left
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
        elif action == 3: # Right
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1]+1)

        # Check if the agent has reached the target
        done = np.all(self.agent_pos == self.target_pos) 
        # Give a reward of 1 if the target is reached, and a small penalty otherwise
        reward = 1.0 if done else -0.01 

        return self._get_observation(), reward, done, {}

    def reset(self):
        # Reset the agent and target positions to random locations
        while True:
            self.agent_pos = self.np_random.randint(self.size, size=2)
            self.target_pos = self.np_random.randint(self.size, size=2)
            if not np.all(self.agent_pos == self.target_pos):
                break

        return self._get_observation()

    def _get_observation(self):
        # Generate the observation as a 3-channel image
        obs = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        obs[self.agent_pos[0], self.agent_pos[1]] = [255, 0, 0] # Agent is red
        obs[self.target_pos[0], self.target_pos[1]] = [0, 255, 0] # Target is green
        return obs

    def render(self):
        # Render the environment to the screen
        if self.viewer is None:
            self.viewer = plt.figure()
            self.img = plt.imshow(self._get_observation())
            plt.show(block=False)
        else:
            self.img.set_data(self._get_observation())
            plt.pause(0.5)
