# Course-task mapping

This file separates the professor's requirements from the implementation choices
made for this project.

## Professor's Intermediate Task 1

The course document asks the team to understand the problem, state the problem
mathematically as an optimization problem, formulate the RL problem, study state,
action, evolution, and reward, and solve a simplified MDP using value iteration or
policy iteration in Python.

Project response:

- `docs/TASK1_REPORT.md` contains the mathematical formulation.
- `src/goal_treasure/exact.py` implements value iteration.
- `scripts/run_task1.py` writes the measured state count and policy.

## Professor's Intermediate Task 2

The course document asks the team to implement or set up a complete simulator of
the system being controlled or optimized through RL. The simulator must output a
sample path of the system under the policy and the report must summarize the
simulator and code.

Project response:

- `src/goal_treasure/env.py` is the complete simulator.
- `src/goal_treasure/uvfa.py` contains the goal-conditioned learner.
- `scripts/train_task2.py` trains and checkpoints the policy.
- `scripts/evaluate_task2.py` tests held-out random seeds.
- `scripts/demo.py` provides the interactive window.
- `docs/TASK2_REPORT.md` explains the simulator and update rule.

## User-specific constraints

- This is an independent project.
- EdgePilot is not imported, modified, copied, or overwritten.
- The target hardware is 8 GB RAM, RTX 3050, and Intel i5 12th gen.
- CUDA is optional; the scripts fall back to CPU.
- No paper result is invented or silently substituted.

