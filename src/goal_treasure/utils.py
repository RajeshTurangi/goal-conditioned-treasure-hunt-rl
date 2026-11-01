"""Configuration, reproducibility, and simple file helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_request(request: str = "auto") -> torch.device:
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports CUDA unavailable")
    if request in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_env(config: dict[str, Any]):
    from .env import TreasureHuntEnv

    return TreasureHuntEnv(
        config["map"],
        max_steps=int(config.get("max_steps", 80)),
        step_reward=float(config.get("step_reward", -0.1)),
        wall_penalty=float(config.get("wall_penalty", -1.0)),
        goal_reward=float(config.get("goal_reward", 10.0)),
        distance_shaping_scale=float(config.get("distance_shaping_scale", 0.0)),
    )

