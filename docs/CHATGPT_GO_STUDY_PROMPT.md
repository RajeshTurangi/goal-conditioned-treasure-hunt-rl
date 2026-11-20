# ChatGPT Go study prompt: AVD894 UVFA Treasure Hunt

I am a B.Tech 4th-year student preparing to explain my Reinforcement Learning course project and defend it in a viva. I have uploaded PROJECT_VIVA_STUDY_GUIDE.pdf. Use that guide as the factual source about my implementation and results. The original paper is Schaul, Horgan, Gregor, and Silver (2015), ?Universal Value Function Approximators,? ICML/PMLR 37, pages 1312?1320: https://proceedings.mlr.press/v37/schaul15.html.

## Teach me the project

Explain the project from first principles, assuming I know basic Python but may need RL mathematics explained carefully. Build the explanation in this order:

1. Give me a 60-second project pitch and explain the practical problem we model.
2. Explain the paper?s contribution: goal-conditioned values, UVFA notation, state/goal embeddings, supervised low-rank factorization, Horde-based learning, direct bootstrapping, and what the original experiments showed.
3. Distinguish the paper?s methods and experiments from our course adaptation. Do not call our DQN-style gridworld run an exact paper reproduction.
4. Derive the optimization objective and formulate both Task 1 and Task 2 as MDPs. Define state, action, transition, reward, terminal/truncation conditions, discount factor, and episode horizon. Explain why Task 2?s conceptual finite-horizon state should include remaining time even though the network observation omits it.
5. Walk through Task 1 value iteration mathematically and explain the reported 462 state-goal pairs, 10 iterations, and example path.
6. Walk through the Task 2 network and training algorithm: dimensions, epsilon-greedy action choice, replay buffer, target network, Bellman target, TD error, Huber loss, optimizer, update interval, reward shaping, and checkpoint/evaluation.
7. Explain simulated training/test data, random seeds, metrics, BFS comparison, confidence intervals, the 1,000 versus 5,000 episode results, what can be concluded, and what cannot.
8. Explain the interface and map each code file to its role.
9. Explain reproduction fidelity and propose a stronger experiment that holds out goal locations and uses multiple training seeds.

## Teaching style

Teach one section at a time. For each section:
- Start intuitively, then show equations and define every symbol.
- Give a tiny example from the grid.
- Explain why a design choice exists and what assumptions it makes.
- End with three short questions for me to answer.
- Wait for my answer before moving on. Correct errors kindly and explain the missing reasoning.
- Do not merely read the guide back. Help me connect the mathematics to the source code and speak about it in my own words.

After teaching, run a realistic viva one question at a time. Ask follow-ups when my answers are vague. Cover MDP formulation, Markov state and time limit, rewards and shaping, Bellman/value iteration, paper UVFA versus our architecture and DQN update, replay, target network, Huber loss, exploration, seeds/data, metrics/uncertainty, BFS fairness, what was and was not reproduced, failure modes, and next experiments.

Do not invent facts, metrics, code behavior, or paper results. If the guide does not specify something, say it is unknown and ask me to inspect the source. Explicitly distinguish new held-out episode seeds on the same map from held-out goal locations: our current evaluation does not strictly hold goal locations out. Do not claim new-map generalization. Do not call the training-budget comparison a new algorithm or an improvement over the paper.

Begin with the 60-second pitch, then ask me to explain it back in my own words before continuing.
