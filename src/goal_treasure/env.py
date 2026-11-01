"""Small deterministic grid simulator used for the course tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ACTIONS: tuple[str, ...] = ("up", "down", "left", "right")
DELTAS: dict[int, tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, np.ndarray | int]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class TreasureHuntEnv:
    """A compact goal-conditioned gridworld with a Gym-like API.

    The observation contains the agent position and goal position as integer
    indices. The neural model additionally receives a flattened obstacle map.
    """

    def __init__(
        self,
        map_rows: Sequence[str],
        *,
        max_steps: int = 80,
        step_reward: float = -0.1,
        wall_penalty: float = -1.0,
        goal_reward: float = 10.0,
        distance_shaping_scale: float = 0.0,
    ) -> None:
        if not map_rows or len({len(row) for row in map_rows}) != 1:
            raise ValueError("map_rows must be a non-empty rectangular map")
        self.map_rows = tuple(map_rows)
        self.height = len(map_rows)
        self.width = len(map_rows[0])
        self.max_steps = max_steps
        self.step_reward = step_reward
        self.wall_penalty = wall_penalty
        self.goal_reward = goal_reward
        self.distance_shaping_scale = distance_shaping_scale
        self.obstacles = np.array(
            [[cell == "#" for cell in row] for row in self.map_rows], dtype=np.bool_
        )
        self.free_cells = tuple(
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if not self.obstacles[r, c]
        )
        if len(self.free_cells) < 2:
            raise ValueError("map must contain at least two free cells")
        self.rng = np.random.default_rng()
        self.position = self.free_cells[0]
        self.goal = self.free_cells[-1]
        self.steps = 0

    @property
    def state_size(self) -> int:
        return self.height * self.width * 2

    @property
    def goal_size(self) -> int:
        return self.height * self.width

    @property
    def action_size(self) -> int:
        return len(ACTIONS)

    def reset(
        self,
        *,
        seed: int | None = None,
        start: tuple[int, int] | None = None,
        goal: tuple[int, int] | None = None,
    ) -> dict[str, np.ndarray | int]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start is None or goal is None:
            chosen = self.rng.choice(len(self.free_cells), size=2, replace=False)
            start = self.free_cells[int(chosen[0])]
            goal = self.free_cells[int(chosen[1])]
        if not self._is_free(start) or not self._is_free(goal) or start == goal:
            raise ValueError("start and goal must be different free cells")
        self.position = start
        self.goal = goal
        self.steps = 0
        return self.observation()

    def observation(self) -> dict[str, np.ndarray | int]:
        return {
            "position": self.position[0] * self.width + self.position[1],
            "goal": self.goal[0] * self.width + self.goal[1],
        }

    def state_vector(self) -> np.ndarray:
        vector = np.zeros(self.state_size, dtype=np.float32)
        position_index = self.position[0] * self.width + self.position[1]
        obstacle_offset = self.height * self.width
        vector[position_index] = 1.0
        vector[obstacle_offset : obstacle_offset + self.height * self.width] = self.obstacles.reshape(-1)
        return vector

    def goal_vector(self) -> np.ndarray:
        vector = np.zeros(self.goal_size, dtype=np.float32)
        goal_index = self.goal[0] * self.width + self.goal[1]
        vector[goal_index] = 1.0
        return vector

    def step(self, action: int) -> StepResult:
        if action not in DELTAS:
            raise ValueError(f"action must be one of {tuple(DELTAS)}")
        dr, dc = DELTAS[action]
        old_distance = self._manhattan(self.position, self.goal)
        candidate = (self.position[0] + dr, self.position[1] + dc)
        self.steps += 1
        if not self._is_free(candidate):
            reward = self.wall_penalty
            candidate = self.position
        else:
            reward = self.step_reward
            new_distance = self._manhattan(candidate, self.goal)
            reward += self.distance_shaping_scale * (old_distance - new_distance)
        self.position = candidate
        terminated = self.position == self.goal
        if terminated:
            reward = self.goal_reward
        truncated = self.steps >= self.max_steps and not terminated
        return StepResult(
            observation=self.observation(),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={"steps": self.steps, "success": terminated},
        )

    def render_ascii(self, path: Iterable[tuple[int, int]] = ()) -> str:
        path_set = set(path)
        rows: list[str] = []
        for r, row in enumerate(self.map_rows):
            chars = list(row)
            for c in range(self.width):
                if (r, c) in path_set and chars[c] == ".":
                    chars[c] = "·"
            if (r, self.goal[1]) == self.goal:
                chars[self.goal[1]] = "G"
            if (r, self.position[1]) == self.position:
                chars[self.position[1]] = "A"
            rows.append("".join(chars))
        return "\n".join(rows)

    def _is_free(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width and not self.obstacles[r, c]

    @staticmethod
    def _manhattan(first: tuple[int, int], second: tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

