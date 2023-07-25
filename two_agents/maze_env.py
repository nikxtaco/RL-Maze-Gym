import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self, size=5, target_pos=None, seed=None):
        super(MazeEnv, self).__init__()

        # Set the seed
        np.random.seed(seed)

        self.size = size
        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Box(low=0, high=3, shape=(size, size, 3), dtype=np.uint8) 

        # Position of the agent and dummy agent
        self.agent_pos = np.array([0, 0])
        self.dummy_agent_pos = np.array([0, 0])

        self.target_pos = target_pos if target_pos else np.random.randint(self.size, size=2)
        self.viewer = None

        self.reset()

    def step(self, action, dummy_action=None):
        if action == 0: # Up
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        elif action == 1: # Down
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0]+1)
        elif action == 2: # Left
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
        elif action == 3: # Right
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1]+1)

        if dummy_action is not None:
            if dummy_action == 0: # Up
                self.dummy_agent_pos[0] = max(0, self.dummy_agent_pos[0]-1)
            elif dummy_action == 1: # Down
                self.dummy_agent_pos[0] = min(self.size-1, self.dummy_agent_pos[0]+1)
            elif dummy_action == 2: # Left
                self.dummy_agent_pos[1] = max(0, self.dummy_agent_pos[1]-1)
            elif dummy_action == 3: # Right
                self.dummy_agent_pos[1] = min(self.size-1, self.dummy_agent_pos[1]+1)

        done = np.all(self.agent_pos == self.target_pos) 
        reward = 1.0 if done else -0.01 

        return self._get_observation(), reward, done, {}

    def reset(self):
        while True:
            self.agent_pos = np.random.randint(self.size, size=2)
            if not np.all(self.agent_pos == self.target_pos):
                break

        self.dummy_agent_pos = np.array([0, 0])  # Initialize dummy agent position to top left

        return self._get_observation()

    def _get_observation(self):
        obs = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        obs[self.agent_pos[0], self.agent_pos[1]] = [255, 0, 0] 
        obs[self.target_pos[0], self.target_pos[1]] = [0, 255, 0] 
        obs[self.dummy_agent_pos[0], self.dummy_agent_pos[1]] = [0, 0, 255]
        return obs

    def render(self):
        if self.viewer is None:
            self.viewer = plt.figure()
            self.img = plt.imshow(self._get_observation())
            plt.show(block=False)
        else:
            self.img.set_data(self._get_observation())
            plt.pause(0.1)
