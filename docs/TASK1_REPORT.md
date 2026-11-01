# Intermediate Task 1 report

## 1. Optimization problem

For a fixed rectangular maze and a selected goal cell, the player must choose a
sequence of moves that reaches the goal while minimizing movement and collision
cost. With step cost `c_step < 0`, the equivalent maximization objective is:

\[
\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

## 2. MDP formulation

- State: `s = (p, g)`, where `p` is the player cell and `g` is the fixed target cell.
- Action: `up`, `down`, `left`, or `right`.
- Transition: a legal move changes the player cell; an illegal move leaves it unchanged.
- Reward: `+10` on reaching the goal, `-0.1` for a legal nonterminal step, and `-1` for a wall collision.
- Terminal condition: the player reaches `g`.
- Discount: `gamma = 0.95` in the supplied configuration.

For a deterministic transition, the Bellman optimality update is:

\[
V_{k+1}(s)=\max_a\left[r(s,a)+\gamma V_k(T(s,a))\right],
\]

with terminal continuation value set to zero.

## 3. Exact Task 1 implementation

`src/goal_treasure/exact.py` enumerates every free-cell/goal pair and applies
value iteration. `scripts/run_task1.py` writes:

- `summary.json`: state count, convergence iterations, tolerance, and an example path.
- `policy.csv`: optimal action/value for every state-goal pair.

The exact solver is deliberately separate from the neural Task 2 agent. This
allows the report to compare an analytically solved small MDP with an approximate
goal-conditioned learner.

