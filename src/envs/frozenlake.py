"""
FrozenLake with pixel observations, for running Delta-IRIS on a toy environment.

Delta-IRIS's tokenizer expects image observations of shape (size, size, 3), dtype uint8.
gym 0.21's FrozenLake only supports 'human' / 'ansi' rendering, so we draw the grid ourselves
with numpy. The frames are clean and cheap to produce, so env stepping adds almost nothing to
the measured runtime.

Uses the gym 0.21 API (same as the rest of this repo):
    reset() -> obs
    step(action) -> (obs, reward, done, info)

Actions (same as gym's FrozenLake): 0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP
"""

from typing import Optional

import gym
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np


# RGB colors for each tile type
TILE_COLORS = {
    b'S': (170, 210, 240),  # start: light blue (frozen)
    b'F': (170, 210, 240),  # frozen: light blue
    b'H': (20, 40, 90),     # hole: dark navy
    b'G': (240, 200, 40),   # goal: gold
}
GRID_LINE_COLOR = (255, 255, 255)
AGENT_COLOR = (220, 40, 40)  # red


class FrozenLakePixels(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, map_name: str = '4x4', is_slippery: bool = True, size: int = 64, desc: Optional[list] = None) -> None:
        self.lake = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=is_slippery)
        self.desc = self.lake.desc  # (nrow, ncol) array of bytes, e.g. b'S', b'F', b'H', b'G'
        self.nrow, self.ncol = self.desc.shape
        self.size = size
        assert size % self.nrow == 0 and size % self.ncol == 0, f'size={size} must be divisible by the grid shape {self.desc.shape}'
        self.cell_h, self.cell_w = size // self.nrow, size // self.ncol

        self.action_space = self.lake.action_space  # Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8)

        self.background = self._draw_background()
        self.state = None

    def _draw_background(self) -> np.ndarray:
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for r in range(self.nrow):
            for c in range(self.ncol):
                y0, x0 = r * self.cell_h, c * self.cell_w
                img[y0:y0 + self.cell_h, x0:x0 + self.cell_w] = TILE_COLORS[self.desc[r, c]]
        # 1-pixel grid lines so cell boundaries are visible
        img[::self.cell_h, :] = GRID_LINE_COLOR
        img[:, ::self.cell_w] = GRID_LINE_COLOR
        return img

    def _render_state(self, state: int) -> np.ndarray:
        img = self.background.copy()
        r, c = divmod(int(state), self.ncol)
        # agent = filled square in the middle half of its cell
        pad_y, pad_x = self.cell_h // 4, self.cell_w // 4
        y0, x0 = r * self.cell_h + pad_y, c * self.cell_w + pad_x
        img[y0:y0 + self.cell_h - 2 * pad_y, x0:x0 + self.cell_w - 2 * pad_x] = AGENT_COLOR
        return img

    def seed(self, seed: Optional[int] = None):
        return self.lake.seed(seed)

    def reset(self) -> np.ndarray:
        self.state = self.lake.reset()
        return self._render_state(self.state)

    def step(self, action: int):
        self.state, reward, done, info = self.lake.step(int(action))
        info = dict(info, state=int(self.state))
        return self._render_state(self.state), float(reward), done, info

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        return self._render_state(self.state)


def make_frozenlake(map_name: str = '4x4', is_slippery: bool = True, size: int = 64, max_episode_steps: int = 100) -> gym.Env:
    # TimeLimit is a gym.Wrapper: MultiProcessEnv reads env_fn().env.action_space, so the outer object must have `.env`
    env = FrozenLakePixels(map_name=map_name, is_slippery=is_slippery, size=size)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
