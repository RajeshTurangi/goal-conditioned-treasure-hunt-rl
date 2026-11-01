# Paper summary

## Citation

Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver, “Universal Value Function Approximators,” Proceedings of the 32nd International Conference on Machine Learning, PMLR 37, 2015, pp. 1312–1320.

Primary source: <https://proceedings.mlr.press/v37/schaul15.html>

## Problem

Ordinary value functions estimate how useful a state is for one fixed objective.
The paper asks whether one approximator can estimate value for many goals and
generalize to goals not observed during training.

## Core formulation

The universal value function is:

\[
V(s,g;\theta),
\]

where `s` is the current state and `g` is a goal descriptor. A goal-conditioned
control policy can choose actions using the value of possible next states for the
current goal.

The paper separates state and goal representations, combines their embeddings,
and updates values from observed rewards. Its experiments include learned value
functions for multiple goals and show generalization to previously unseen goals.

## Course-project adaptation

The Task 1/Task 2 environment here is a small grid-based treasure hunt. This is
an explicit adaptation chosen to satisfy the course requirement for a complete,
inspectable simulator. It is not presented as an exact recreation of every
experimental detail in the original paper.

The implementation preserves the paper’s central research question:

> Can one value model condition on a goal and act successfully for new goals?

The Task 2 agent uses a factorized state encoder and goal encoder followed by a
Q-value head. The Q-learning update and replay/target-network machinery make the
adaptation executable on the student hardware.

