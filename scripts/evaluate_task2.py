"""Evaluate a Task 2 checkpoint on deterministic but unseen random seeds."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal_treasure.env import DELTAS
from goal_treasure.uvfa import load_checkpoint
from goal_treasure.utils import device_from_request, load_json, make_env, save_json


def shortest_path_action(env) -> int | None:
    queue = deque([(env.position, [])])
    visited = {env.position}
    while queue:
        cell, actions = queue.popleft()
        if cell == env.goal:
            return actions[0] if actions else None
        for action, (dr, dc) in DELTAS.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt not in visited and env._is_free(nxt):
                visited.add(nxt)
                queue.append((nxt, actions + [action]))
    return None


def run_episode(env, network, device, seed: int, learned: bool) -> dict[str, float | int]:
    env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(env.max_steps):
        if learned:
            with torch.no_grad():
                q_values = network(torch.from_numpy(env.state_vector()).unsqueeze(0).to(device), torch.from_numpy(env.goal_vector()).unsqueeze(0).to(device))
            action = int(q_values.argmax(dim=1).item())
        else:
            action = shortest_path_action(env)
            if action is None:
                break
        result = env.step(action)
        total_reward += result.reward
        if result.terminated or result.truncated:
            break
    return {"seed": seed, "success": int(env.position == env.goal), "steps": env.steps, "return": total_reward}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = load_json(args.config)
    device = device_from_request(args.device)
    env = make_env(config)
    network, metadata = load_checkpoint(args.checkpoint, device)
    learned = [run_episode(env, network, device, 100_000 + i, True) for i in range(args.episodes)]
    baseline = [run_episode(env, network, device, 100_000 + i, False) for i in range(args.episodes)]
    result = {
        "checkpoint_metadata": metadata,
        "episodes": args.episodes,
        "learned": {
            "success_rate": float(np.mean([row["success"] for row in learned])),
            "mean_steps_successful": float(np.mean([row["steps"] for row in learned if row["success"]])) if any(row["success"] for row in learned) else None,
            "mean_return": float(np.mean([row["return"] for row in learned])),
        },
        "shortest_path_reference": {
            "success_rate": float(np.mean([row["success"] for row in baseline])),
            "mean_steps": float(np.mean([row["steps"] for row in baseline])),
            "mean_return": float(np.mean([row["return"] for row in baseline])),
        },
        "learned_episode_rows": learned,
    }
    save_json(args.output, result)
    print(f"learned success rate: {result['learned']['success_rate']:.3f}")
    print(f"shortest-path reference success rate: {result['shortest_path_reference']['success_rate']:.3f}")
    print(f"saved evaluation to {args.output}")


if __name__ == "__main__":
    main()

