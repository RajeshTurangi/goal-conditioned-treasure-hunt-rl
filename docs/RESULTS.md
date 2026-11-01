# Results log

This file is intentionally a log, not a place for invented numbers.

## Task 1

Measured on 2026-11-02 with `configs/task1_small.json`:

- Free cells: 22
- State-goal pairs: 462
- Actions: 4
- Value-iteration convergence: 10 iterations
- Final delta: 0.0 at the configured floating-point tolerance
- Example path length: 9 moves

## Task 2

The following numbers are measured and are not paper-reported numbers.

### Unshaped 80-episode smoke run

- Device: CUDA (`torch 2.5.1+cu121`)
- Held-out evaluation: 100 random seeds
- Learned success rate: 10.0%
- Shortest-path reference success rate: 100.0%
- Interpretation: under-trained diagnostic run; not used as a claim of success.

### Shaped 200-episode Task 2 run

- Device: CUDA (`torch 2.5.1+cu121`)
- Map: `configs/task2_default.json`
- Training seed: 2026
- Evaluation: 200 held-out seeds (`100000`–`100199`)
- Learned success rate: 52.5%
- Learned mean steps among successful episodes: 4.4095
- Learned mean return: 3.0230
- Shortest-path reference success rate: 100.0%
- Shortest-path reference mean steps: 4.4
- Interpretation: measurable learning on the adapted simulator; further training and multi-seed evaluation are required before making a stronger performance claim.

For later runs, record:

- Git commit
- Python, NumPy, and PyTorch versions
- Device and CUDA availability
- Map/configuration
- Number of episodes
- Random seeds
- Training time
- Checkpoint size
- Learned success rate and mean return
- Shortest-path reference success rate and mean return

Do not compare against the original paper’s numbers unless the relevant
experimental conditions have actually been reproduced.

