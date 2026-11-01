"""A compact UVFA-style Q-learning agent for Intermediate Task 2."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import numpy as np
import torch
from torch import nn


class UVFAQNetwork(nn.Module):
    """Factorized state/goal encoders followed by an action-value head."""

    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 128, embedding_dim: int = 64) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim), nn.ReLU())
        self.goal_encoder = nn.Sequential(nn.Linear(goal_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim), nn.ReLU())
        self.q_head = nn.Sequential(nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        state_embedding = self.state_encoder(state)
        goal_embedding = self.goal_encoder(goal)
        return self.q_head(torch.cat((state_embedding, goal_embedding), dim=-1))


@dataclass
class Transition:
    state: np.ndarray
    goal: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_goal: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.data: Deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self.data.append(transition)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(list(self.data), batch_size)


def select_action(
    network: UVFAQNetwork,
    state: np.ndarray,
    goal: np.ndarray,
    epsilon: float,
    action_dim: int,
    device: torch.device,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return rng.randrange(action_dim)
    with torch.no_grad():
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        goal_tensor = torch.from_numpy(goal).unsqueeze(0).to(device)
        return int(network(state_tensor, goal_tensor).argmax(dim=1).item())


def optimize_batch(
    online: UVFAQNetwork,
    target: UVFAQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: list[Transition],
    gamma: float,
    device: torch.device,
) -> float:
    states = torch.from_numpy(np.stack([item.state for item in batch])).to(device)
    goals = torch.from_numpy(np.stack([item.goal for item in batch])).to(device)
    actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.stack([item.next_state for item in batch])).to(device)
    next_goals = torch.from_numpy(np.stack([item.next_goal for item in batch])).to(device)
    done = torch.tensor([item.done for item in batch], dtype=torch.float32, device=device)

    current_q = online(states, goals).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(next_states, next_goals).max(dim=1).values
        target_q = rewards + gamma * (1.0 - done) * next_q
    loss = nn.functional.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(path: str | Path, network: UVFAQNetwork, metadata: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": network.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[UVFAQNetwork, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(checkpoint["metadata"])
    network = UVFAQNetwork(
        int(metadata["state_dim"]),
        int(metadata["goal_dim"]),
        int(metadata["action_dim"]),
        int(metadata["hidden_dim"]),
        int(metadata["embedding_dim"]),
    ).to(device)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    return network, metadata

