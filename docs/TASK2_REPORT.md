# Intermediate Task 2 report

## Simulator

`src/goal_treasure/env.py` implements the complete controlled system. It
provides reset, step, observation, vector encoding, ASCII rendering, rewards,
termination, truncation, and reproducible random seeds.

## Neural UVFA adaptation

`src/goal_treasure/uvfa.py` implements:

1. A state encoder.
2. A separate goal encoder.
3. Concatenated state/goal embeddings.
4. An action-value head producing one value per movement action.
5. Experience replay.
6. A target network.
7. Huber temporal-difference loss.

For a transition `(s, g, a, r, s', g')`, the training target is:

\[
y=r+\gamma(1-d)\max_{a'} Q_{\text{target}}(s',g',a'),
\]

and the optimized loss is:

\[
L=\operatorname{Huber}\left(Q_{\text{online}}(s,g,a),y\right).
\]

The current Task 2 configuration adds a small Manhattan-distance potential
shaping term to make the first simulator learnable within course-scale smoke
runs. This is an explicit adaptation, not a claim about the original paper's
reward. The exact Task 1 configuration keeps the unshaped reward defined in
`TASK1_REPORT.md`.

## Training and evaluation

`scripts/train_task2.py` records seed, device, software version, episode returns,
successes, steps, epsilon, and loss count in `training.json` and saves a checkpoint.

`scripts/evaluate_task2.py` evaluates on seeds that are separate from the training
seeds and compares the learned policy with a shortest-path reference controller.

## Demonstration

`scripts/demo.py` opens a Tkinter window. It displays the maze, the agent, the
current goal, the path, the episode status, and a button for each next model step.

