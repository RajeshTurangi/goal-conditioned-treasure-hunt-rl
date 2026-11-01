"""Interactive Tkinter demonstration for the Task 2 simulator."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goal_treasure.uvfa import load_checkpoint
from goal_treasure.utils import device_from_request, load_json, make_env


class TreasureWindow:
    def __init__(self, env, network, device) -> None:
        self.env = env
        self.network = network
        self.device = device
        self.root = tk.Tk()
        self.root.title("Goal-Conditioned Treasure Hunt")
        self.cell_size = 52
        self.canvas = tk.Canvas(self.root, width=env.width * self.cell_size, height=env.height * self.cell_size, bg="white")
        self.canvas.pack(padx=10, pady=10)
        self.info = tk.Label(self.root, anchor="w", justify="left", font=("Consolas", 10))
        self.info.pack(fill="x", padx=10)
        controls = tk.Frame(self.root)
        controls.pack(pady=(0, 10))
        tk.Button(controls, text="New random episode", command=self.new_episode).pack(side="left", padx=4)
        tk.Button(controls, text="Next RL step", command=self.step).pack(side="left", padx=4)
        tk.Button(controls, text="Quit", command=self.root.destroy).pack(side="left", padx=4)
        self.path: list[tuple[int, int]] = []
        self.new_episode()

    def new_episode(self) -> None:
        self.env.reset()
        self.path = [self.env.position]
        self.draw("new episode")

    def step(self) -> None:
        if self.env.position == self.env.goal or self.env.steps >= self.env.max_steps:
            return
        with torch.no_grad():
            q_values = self.network(torch.from_numpy(self.env.state_vector()).unsqueeze(0).to(self.device), torch.from_numpy(self.env.goal_vector()).unsqueeze(0).to(self.device))
        action = int(q_values.argmax(dim=1).item())
        result = self.env.step(action)
        self.path.append(self.env.position)
        self.draw(f"action={action} reward={result.reward:+.2f}")

    def draw(self, status: str) -> None:
        self.canvas.delete("all")
        path = set(self.path)
        for r in range(self.env.height):
            for c in range(self.env.width):
                x0, y0 = c * self.cell_size, r * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                if self.env.obstacles[r, c]:
                    fill = "#30343b"
                elif (r, c) in path:
                    fill = "#d9efff"
                else:
                    fill = "#ffffff"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#aab2bd")
        gr, gc = self.env.goal
        self.canvas.create_oval(gc * self.cell_size + 12, gr * self.cell_size + 12, (gc + 1) * self.cell_size - 12, (gr + 1) * self.cell_size - 12, fill="#38a169", outline="")
        ar, ac = self.env.position
        self.canvas.create_oval(ac * self.cell_size + 15, ar * self.cell_size + 15, (ac + 1) * self.cell_size - 15, (ar + 1) * self.cell_size - 15, fill="#2563eb", outline="")
        outcome = "SUCCESS" if self.env.position == self.env.goal else "running"
        self.info.configure(text=f"Status: {status} | Episode: {outcome} | Steps: {self.env.steps}/{self.env.max_steps} | Blue=agent  Green=goal  Grey=wall")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    env = make_env(config)
    device = device_from_request(args.device)
    if args.no_model:
        from goal_treasure.uvfa import UVFAQNetwork

        network = UVFAQNetwork(
            env.state_size,
            env.goal_size,
            env.action_size,
            int(config.get("hidden_dim", 128)),
            int(config.get("embedding_dim", 64)),
        ).to(device)
    elif args.checkpoint:
        network, _ = load_checkpoint(args.checkpoint, device)
    else:
        raise SystemExit("provide --checkpoint or use --no-model")
    TreasureWindow(env, network, device).run()


if __name__ == "__main__":
    main()

