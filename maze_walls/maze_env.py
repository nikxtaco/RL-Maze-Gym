import numpy as np
import random
import gym
from gym import spaces
import pygame

class MazeEnv(gym.Env):
    def __init__(self, size=5, target_pos=None, agent_id=0, num_walls=0, seed=None):
        super(MazeEnv, self).__init__()
        self.seed(seed)
        self.size = size
        self.agent_pos = None
        self.state = None
        self.target_pos = target_pos
        self.agent_id = agent_id
        self.walls = []
        self.num_walls = num_walls
        self.reset()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(size, size, 3))

        # initialize pygame for rendering
        pygame.init()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.state = np.zeros((self.size, self.size, 3))

        if self.target_pos is None:  # target position is not set, choose randomly
            self.target_pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        
        self.state[self.target_pos[0], self.target_pos[1], 0] = 1

        self.walls = []
        for _ in range(self.num_walls):
            while True:
                wall = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
                if not np.array_equal(wall, self.agent_pos) and not np.array_equal(wall, self.target_pos):
                    self.state[wall[0], wall[1], 1] = 1
                    self.walls.append(wall)
                    break

        return self.state

    def step(self, action):
        if action == 0:   # up
            next_pos = [max(0, self.agent_pos[0]-1), self.agent_pos[1]]
        elif action == 1: # right
            next_pos = [self.agent_pos[0], min(self.size-1, self.agent_pos[1]+1)]
        elif action == 2: # down
            next_pos = [min(self.size-1, self.agent_pos[0]+1), self.agent_pos[1]]
        elif action == 3: # left
            next_pos = [self.agent_pos[0], max(0, self.agent_pos[1]-1)]

        if not any(np.array_equal(next_pos, wall) for wall in self.walls):
            self.agent_pos = next_pos

        self.state = np.zeros((self.size, self.size, 3))

        self.state[self.agent_pos[0], self.agent_pos[1], 2] = 0.5  # agent's position
        for wall in self.walls:  # walls
            self.state[wall[0], wall[1], 1] = 1

        if self.target_pos is not None:  # target position
            self.state[self.target_pos[0], self.target_pos[1], 0] = 1

        done = np.array_equal(self.agent_pos, self.target_pos)
        reward = float(done) - 0.1

        return self.state, reward, done, {}

    def render(self, window=None):
        if window is None: return  # Skip rendering if no window provided
        window.fill((255,255,255))
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i,j,2] == 0.5: # agent
                    color = (0,0,255)
                elif self.state[i,j,1] == 1: # wall
                    color = (0,0,0)
                elif self.state[i,j,0] == 1: # goal
                    color = (255,0,0)
                else:
                    color = (255,255,255)
                pygame.draw.rect(window, color, pygame.Rect(j*100,i*100,100,100))
        pygame.display.flip()

from gym.envs.registration import register

register(
    id='Maze-v0',
    entry_point='maze_env:MazeEnv',
)
