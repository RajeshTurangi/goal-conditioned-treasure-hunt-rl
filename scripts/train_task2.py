"""Train the Task 2 neural UVFA-style Q-network."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal_treasure.uvfa import ReplayBuffer, Transition, UVFAQNetwork, optimize_batch, save_checkpoint, select_action
from goal_treasure.utils import device_from_request, load_json, make_env, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = load_json(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    device = device_from_request(args.device)
    rng = random.Random(seed)
    env = make_env(config)
    online = UVFAQNetwork(env.state_size, env.goal_size, env.action_size, int(config["hidden_dim"]), int(config["embedding_dim"])).to(device)
    target = UVFAQNetwork(env.state_size, env.goal_size, env.action_size, int(config["hidden_dim"]), int(config["embedding_dim"])).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=float(config["learning_rate"]))
    replay = ReplayBuffer(int(config["replay_capacity"]))
    losses: list[float] = []
    episode_rows: list[dict[str, float | int]] = []
    global_step = 0
    epsilon_start = float(config["epsilon_start"])
    epsilon_end = float(config["epsilon_end"])
    epsilon_decay_steps = int(config["epsilon_decay_steps"])

    for episode in range(args.episodes):
        env.reset(seed=seed + episode)
        state = env.state_vector()
        goal = env.goal_vector()
        total_reward = 0.0
        success = 0
        epsilon = epsilon_start
        for _ in range(env.max_steps):
            fraction = min(global_step / max(epsilon_decay_steps, 1), 1.0)
            epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
            action = select_action(online, state, goal, epsilon, env.action_size, device, rng)
            result = env.step(action)
            next_state = env.state_vector()
            next_goal = env.goal_vector()
            done = result.terminated or result.truncated
            replay.append(Transition(state, goal, action, result.reward, next_state, next_goal, float(done)))
            state, goal = next_state, next_goal
            total_reward += result.reward
            success = int(result.terminated)
            global_step += 1
            if len(replay) >= int(config["warmup_steps"]):
                loss = optimize_batch(online, target, optimizer, replay.sample(int(config["batch_size"]), rng), float(config["gamma"]), device)
                losses.append(loss)
            if global_step % int(config["target_update_interval"]) == 0:
                target.load_state_dict(online.state_dict())
            if done:
                break
        episode_rows.append({"episode": episode + 1, "return": total_reward, "success": success, "steps": env.steps, "epsilon": epsilon})
        if (episode + 1) % max(1, args.episodes // 10) == 0:
            recent = episode_rows[-max(1, args.episodes // 10) :]
            print(f"episode={episode + 1}/{args.episodes} success={np.mean([r['success'] for r in recent]):.3f} loss={np.mean(losses[-100:]) if losses else float('nan'):.4f} device={device}")

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "state_dim": env.state_size,
        "goal_dim": env.goal_size,
        "action_dim": env.action_size,
        "hidden_dim": int(config["hidden_dim"]),
        "embedding_dim": int(config["embedding_dim"]),
        "map": config["map"],
        "seed": seed,
        "episodes": args.episodes,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    save_checkpoint(args.output / "checkpoint.pt", online, metadata)
    save_json(args.output / "training.json", {"metadata": metadata, "loss_count": len(losses), "episode_rows": episode_rows})
    print(f"Saved checkpoint to {args.output / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()

