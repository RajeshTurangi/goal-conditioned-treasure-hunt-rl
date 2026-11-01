# Reproduction and deviation plan

## Paper-faithful target

The faithful target is the paper’s UVFA concept: condition a value model on both
state and goal and evaluate transfer to goals not present in the training subset.

## Task 1

The finite grid is solved exactly with value iteration. This is a course-required
mathematical reference, not a claim about the paper’s large-scale experiments.

## Task 2

The simulator and neural UVFA adaptation are intentionally small enough for an
8 GB RAM / RTX 3050 machine. The adaptation changes the environment to a custom
treasure-hunt grid, uses an action-value head for control, and uses a modern
PyTorch implementation with a target network and replay buffer. The current
Task 2 smoke configuration also uses a small Manhattan-distance potential
shaping term. This is recorded as a deviation because the course task requires
a working simulator; it is not silently counted as an original-paper result.

## What is not claimed

- No original paper result is copied into the generated result files.
- A smoke run is not evidence of reproduction of the paper’s reported performance.
- The custom grid simulator is not the original paper environment.
- No extension result is reported until it is separately implemented and measured.

