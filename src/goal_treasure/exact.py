"""Exact dynamic-programming solution for Intermediate Task 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .env import DELTAS

Cell = tuple[int, int]
State = tuple[Cell, Cell]


@dataclass(frozen=True)
class ValueIterationResult:
    values: Mapping[State, float]
    policy: Mapping[State, int]
    iterations: int
    delta: float


def value_iteration(
    obstacles: np.ndarray,
    *,
    gamma: float = 0.95,
    step_reward: float = -0.1,
    wall_penalty: float = -1.0,
    goal_reward: float = 10.0,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> ValueIterationResult:
    """Solve the fixed-map, fixed-goal MDP for every free start/goal pair."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    free: tuple[Cell, ...] = tuple(
        (r, c)
        for r in range(obstacles.shape[0])
        for c in range(obstacles.shape[1])
        if not obstacles[r, c]
    )
    states: tuple[State, ...] = tuple((position, goal) for position in free for goal in free if position != goal)
    values: dict[State, float] = {state: 0.0 for state in states}
    policy: dict[State, int] = {state: 0 for state in states}

    for iteration in range(1, max_iterations + 1):
        new_values: dict[State, float] = {}
        delta = 0.0
        for position, goal in states:
            candidates: list[float] = []
            for action, (dr, dc) in DELTAS.items():
                next_position = (position[0] + dr, position[1] + dc)
                if not _is_free(obstacles, next_position):
                    next_position = position
                    reward = wall_penalty
                elif next_position == goal:
                    reward = goal_reward
                else:
                    reward = step_reward
                continuation = 0.0 if next_position == goal else values.get((next_position, goal), 0.0)
                candidates.append(reward + gamma * continuation)
            best_action = int(np.argmax(candidates))
            new_values[(position, goal)] = candidates[best_action]
            policy[(position, goal)] = best_action
            delta = max(delta, abs(new_values[(position, goal)] - values[(position, goal)]))
        values = new_values
        if delta < tolerance:
            return ValueIterationResult(values, policy, iteration, delta)
    return ValueIterationResult(values, policy, max_iterations, delta)


def follow_policy(
    policy: Mapping[State, int],
    start: Cell,
    goal: Cell,
    obstacles: np.ndarray,
    *,
    max_steps: int = 200,
) -> list[Cell]:
    """Return the path produced by an exact policy."""
    position = start
    path = [position]
    for _ in range(max_steps):
        if position == goal:
            break
        action = policy[(position, goal)]
        dr, dc = DELTAS[action]
        candidate = (position[0] + dr, position[1] + dc)
        position = candidate if _is_free(obstacles, candidate) else position
        path.append(position)
    return path


def _is_free(obstacles: np.ndarray, cell: Cell) -> bool:
    r, c = cell
    return 0 <= r < obstacles.shape[0] and 0 <= c < obstacles.shape[1] and not obstacles[r, c]

