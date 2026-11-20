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

### Submission 1,000-episode Task 2 run

- Device: CUDA (`torch 2.5.1+cu121`)
- Map: `configs/task2_default.json`
- Training seed: 2026
- Training episodes: 1,000
- Measured training wall time: approximately 52.6 seconds in the notebook run
- Evaluation: 500 held-out seeds (`100000`–`100499`)
- Learned success rate: 98.6%
- Learned mean steps among successful episodes: 4.5456
- Learned mean return: 9.9012
- Shortest-path reference success rate: 100.0%
- Shortest-path reference mean steps: 4.28
- Checkpoint: `artifacts/task2_submission_1000/checkpoint.pt`
- Evaluation file: `artifacts/task2_submission_1000/evaluation.json`
- Interpretation: strong performance on the fixed adapted maze; not a claim of reproducing the original paper's experimental numbers.

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



## Extended Task 2 training-budget comparison

Measured on the same fixed map and the same 500 held-out seeds (100000-100499),
with all other settings unchanged:

| Training episodes | Successes | Success rate (Wilson 95% CI) | Mean steps when successful (bootstrap 95% CI) | Mean return |
|---:|---:|---:|---:|---:|
| 1,000 | 493/500 | 98.6% (97.14-99.32%) | 4.546 (4.349-4.744) | 9.9012 |
| 5,000 | 500/500 | 100.0% (99.24-100.00%) | 4.284 (4.104-4.466) | 10.1378 |
| Shortest-path reference | 500/500 | 100.0% | 4.280 moves | 10.1382 |

The 5,000-episode training log contains 28,770 environment transitions and
28,571 optimizer updates. Its last 500 training episodes succeeded at a rate of
100%. On the matched evaluation sample it improved the observed result over the
1,000-episode checkpoint. This comparison changes the training budget only; it
does not establish better generalization beyond this fixed maze, and both
checkpoints use the same training seed. The shortest-path controller knows the
maze and is a reference, not an RL baseline.

The Wilson intervals are for the per-episode success proportion. The path-length
intervals are percentile bootstrap intervals over successful episodes using
20,000 resamples with bootstrap seed 2026. Since the path-length intervals
overlap, do not claim statistical separation of those two conditional means.
The 100% observed success rate still has a nonzero uncertainty interval.

Artifacts (ignored by Git) are in artifacts/task2_submission_1000/ and
artifacts/task2_extended_5000/. The report figure is tracked at
docs/task2_training_comparison.png.

