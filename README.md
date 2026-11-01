# Goal-Conditioned Treasure Hunt

Independent AV D894 course-project implementation for Tasks 1 and 2 based on:

> Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. “Universal Value Function Approximators.” ICML 2015.

Primary paper: <https://proceedings.mlr.press/v37/schaul15.html>

This project turns the paper’s goal-conditioned value-function idea into a small,
observable treasure-hunt game. It has two deliberately separate parts:

- `task1_exact/`: a finite-grid MDP and exact value iteration, as required by Intermediate Task 1.
- `task2_uvfa/`: a neural goal-conditioned value approximator, replay buffer, simulator, evaluation scripts, and a Tkinter demo, as required by Intermediate Task 2.

The game environment is a course-project adaptation. It is not claimed to be the
original experimental environment from the paper. The paper reproduction target is
the central UVFA idea: one value model conditions on both state and goal and should
generalize to goals that were not used in training.

## Current scope

Implemented through Intermediate Task 2:

1. Mathematical MDP definition and exact value iteration on a small fixed maze.
2. A complete Python simulator controlled by a trained neural UVFA-style Q-network.
3. Randomized start/goal evaluation, baseline comparison, checkpointing, plots/CSV summaries,
   and an interactive demo.

No extension claim is made yet. Machine-learning extensions and larger/randomized maps are
reserved for later work.

## Quick start

Create an environment and install this project locally:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the exact Task 1 solver:

```powershell
python scripts/run_task1.py --config configs/task1_small.json --output artifacts/task1
```

Run the short Task 2 smoke training run:

```powershell
python scripts/train_task2.py --config configs/task2_default.json --episodes 80 --output artifacts/task2_smoke
python scripts/evaluate_task2.py --config configs/task2_default.json --checkpoint artifacts/task2_smoke/checkpoint.pt --episodes 100 --output artifacts/task2_smoke/evaluation.json
```

Open the simulator window after a checkpoint exists:

```powershell
python scripts/demo.py --config configs/task2_default.json --checkpoint artifacts/task2_smoke/checkpoint.pt
```

The demo needs a desktop session. Use `--no-model` to inspect the simulator with a
shortest-path controller before training.

Run automated tests:

```powershell
python -m unittest discover -s tests -v
```

## Hardware and runtime policy

The scripts select CUDA only when PyTorch reports it as available; otherwise they use CPU.
The smoke run is intentionally small. A longer run must be planned from measured smoke-run
throughput before starting it. Do not treat the generated smoke result as a reproduction of
the paper’s reported numbers.

## Project map

```text
configs/                 Reproducible Task 1/Task 2 settings
docs/                    Paper summary, task reports, deviations, result log, AI log
scripts/                 Task 1, training, evaluation, and GUI entry points
src/goal_treasure/       Environment, exact solver, UVFA network, replay buffer, utilities
tests/                   Automated smoke/unit tests
artifacts/               Generated locally; ignored by Git
```

## Citation and attribution

```bibtex
@inproceedings{schaul2015uvfa,
  title={Universal Value Function Approximators},
  author={Schaul, Tom and Horgan, Daniel and Gregor, Karol and Silver, David},
  booktitle={Proceedings of the 32nd International Conference on Machine Learning},
  pages={1312--1320},
  year={2015}
}
```

The paper is licensed and cited separately from this original course-project code.

