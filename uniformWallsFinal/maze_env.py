import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self, size=5, seed=None):
        super(MazeEnv, self).__init__()
        self.size = size | 1  # Ensure odd size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=(size, size), dtype=int)
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def _recursive_backtracking(self, x, y, maze):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.np_random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 1 <= nx < self.size - 1 and 1 <= ny < self.size - 1 and maze[nx][ny] == 1:
                maze[nx - dx][ny - dy] = 0
                maze[nx][ny] = 0
                self._recursive_backtracking(nx, ny, maze)

    def _generate_maze(self):
        maze = np.ones((self.size, self.size))
        for i in range(1, self.size, 2):
            maze[i, :] = 0
            maze[:, i] = 0
        for _ in range(self.size * self.size // 4):
            rx, ry = self.np_random.integers(1, self.size - 1, 2)
            maze[rx][ry] = 0

        x, y = self.np_random.integers(1, self.size-1, 2)
        maze[x][y] = 0
        self._recursive_backtracking(x, y, maze)

        self.agent_pos = (self.np_random.integers(1, self.size-1), 0)
        self.target_pos = self.np_random.choice(np.argwhere(maze == 0))
        maze[self.target_pos[0]][self.target_pos[1]] = 2

        return maze

    def reset(self):
        self.maze = self._generate_maze()
        self.agent_pos = (self.size-2, 0)
        self.maze[self.agent_pos] = 2
        self.maze[self.target_pos[0]][self.target_pos[1]] = 2
        self.steps = 0
        return self.maze.copy()

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)

        if self.maze[x, y] == 1:  # Hit wall
            reward = -1
        else:
            self.maze[self.agent_pos] = 0
            self.agent_pos = (x, y)
            self.maze[self.agent_pos] = 2
            reward = 0

        if self.agent_pos == tuple(self.target_pos):  # Reached target
            reward = 10
            done = True
        else:
            done = False

        self.steps += 1
        if self.steps >= 100:  # Max steps reached
            done = True

        return self.maze.copy(), reward, done, {}

    def render(self, mode='human'):
        colored_maze = np.zeros((self.size + 2, self.size + 2, 3)) + [0.6, 0.3, 0]  # Brown border
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.agent_pos:
                    colored_maze[i+1, j+1] = [0, 1, 0]  # Green agent
                elif (i, j) == tuple(self.target_pos):
                    colored_maze[i+1, j+1] = [1, 0, 0]  # Red target
                elif self.maze[i, j] == 1:
                    colored_maze[i+1, j+1] = [0, 0, 0]  # Black walls
                else:
                    colored_maze[i+1, j+1] = [1, 1, 1]  # White empty cells

        plt.imshow(colored_maze)
        plt.axis('off')
        plt.pause(0.5)  # To slow down the rendering
        plt.draw()
