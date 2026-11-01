"""Run Intermediate Task 1: formulate and solve the finite MDP exactly."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal_treasure.exact import follow_policy, value_iteration
from goal_treasure.utils import load_json, make_env, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    env = make_env(config)
    result = value_iteration(
        env.obstacles,
        gamma=float(config["gamma"]),
        step_reward=float(config["step_reward"]),
        wall_penalty=float(config["wall_penalty"]),
        goal_reward=float(config["goal_reward"]),
        tolerance=float(config["tolerance"]),
        max_iterations=int(config["max_iterations"]),
    )
    start, goal = env.free_cells[0], env.free_cells[-1]
    path = follow_policy(result.policy, start, goal, env.obstacles)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output / "summary.json",
        {
            "task": "Intermediate Task 1",
            "state_count": len(result.values),
            "free_cell_count": len(env.free_cells),
            "action_count": env.action_size,
            "iterations": result.iterations,
            "final_delta": result.delta,
            "gamma": config["gamma"],
            "example_start": list(start),
            "example_goal": list(goal),
            "example_path_length": len(path) - 1,
            "example_path": [list(cell) for cell in path],
        },
    )
    with (args.output / "policy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_row", "start_col", "goal_row", "goal_col", "action", "value"])
        for (start_cell, goal_cell), action in sorted(result.policy.items()):
            writer.writerow([*start_cell, *goal_cell, action, result.values[(start_cell, goal_cell)]])
    print(f"Solved {len(result.values)} state-goal pairs in {result.iterations} iterations")
    print(f"Example path length: {len(path) - 1}")
    print("Example policy path:")
    print(" -> ".join(str(cell) for cell in path))


if __name__ == "__main__":
    main()

