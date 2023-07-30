import gym
from gym import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self, size=10, target_pos=None):
        super(MazeEnv, self).__init__()

        self.maze_size = size
        self.agent_position = [0, 0]
        self.target_position = target_pos if target_pos else [size - 1, size - 1]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([size-1, size-1]))

        self.maze = np.zeros((size, size))

        # Add walls to the maze
        self.maze[1, 1:4] = 1
        self.maze[1:4, 1] = 1
        self.maze[1:4, 3] = 1
        self.maze[3, 1:4] = 1

    def step(self, action):
        old_position = self.agent_position.copy()

        if action == 0:
            self.agent_position[0] += 1
        elif action == 1:
            self.agent_position[0] -= 1
        elif action == 2:
            self.agent_position[1] += 1
        elif action == 3:
            self.agent_position[1] -= 1

        self.agent_position = np.clip(self.agent_position, 0, self.maze_size-1)

        # If the new position is a wall, stay at the old position
        if self.maze[tuple(self.agent_position)] == 1:
            self.agent_position = old_position

        if tuple(self.agent_position) == tuple(self.target_position):
            reward = 1
            done = True
        else:
            reward = -0.01
            done = False

        return self.agent_position, reward, done, {}

    def reset(self):
        self.agent_position = [0, 0]
        return self.agent_position

    def render(self):
        maze_to_draw = np.copy(self.maze)
        maze_to_draw[tuple(self.agent_position)] = 2
        print(maze_to_draw)
