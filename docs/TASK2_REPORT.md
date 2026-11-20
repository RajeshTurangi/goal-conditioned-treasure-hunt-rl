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



## Longer training comparison

We kept the simulator, network, optimizer, reward shaping, epsilon schedule,
and seed fixed, and increased the training budget from 1,000 to 5,000 episodes.
The extended run collected 28,770 transitions. Both checkpoints were evaluated
on the same 500 unseen random start/goal seeds.

- 1,000 episodes: 493/500 successes (98.6%); mean successful length 4.546 moves;
  mean return 9.9012.
- 5,000 episodes: 500/500 successes (100%); mean successful length 4.284 moves;
  mean return 10.1378.
- Shortest-path reference: 500/500 successes; 4.280 mean moves; mean return 10.1382.

For the 5,000-episode result, the Wilson 95% interval for success is
99.24-100.00%. The percentile-bootstrap 95% interval for successful path length
is 4.104-4.466 moves (20,000 resamples, seed 2026). The 1,000-episode intervals
and comparison plot are in docs/RESULTS.md and
docs/task2_training_comparison.png. This evidence supports a training-budget
comparison on this one map; it does not establish generalization to different
maps or quantify variation across independent training seeds.

## Demonstration window

Run scripts/demo.py with the 5,000-episode checkpoint. The Tkinter interface
shows the agent, goal, obstacles, and visited path. Next RL step advances one
action, Auto-play animates the learned policy, Pause stops it, and New random
episode samples a new start and goal. The panel reports the selected action,
reward, current return, and number of moves. This runs locally with Python's
built-in Tkinter; it does not need a dataset or a web service.
