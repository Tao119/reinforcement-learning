# Reinforcement Learning from Scratch

Pure-NumPy implementations of core RL algorithms on a simple 4×4 GridWorld environment.

## Algorithms

| Chapter | Algorithm | File | Key Idea |
|---------|-----------|------|----------|
| ch01 | Multi-Armed Bandit | `ch01_bandit/bandit.py` | Epsilon-Greedy, UCB1, Thompson Sampling on exploration-exploitation trade-off |
| ch02 | Value Iteration | `ch02_mdp/value_iteration.py` | Dynamic programming; V(s) ← max_a Σ P(s'\|s,a)[R + γV(s')] |
| ch03 | Q-Learning | `ch03_qlearning/qlearning.py` | Tabular TD control; Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) − Q(s,a)] |
| ch04 | DQN | `ch04_dqn/dqn.py` | Neural Q-function (pure NumPy), experience replay, target network |
| ch05 | REINFORCE | `ch05_policy_gradient/reinforce.py` | Policy gradient with variance-reducing baseline |

## Environment

`common/env.py` provides two environments:

- **GridWorld(4×4)** — goal at (3,3), hole at (1,1), 4 actions (↑↓←→)
- **MultiArmedBandit(10 arms)** — fixed Bernoulli reward probabilities

## Quick Start

```bash
python ch01_bandit/bandit.py         # → bandit_results.png
python ch02_mdp/value_iteration.py
python ch03_qlearning/qlearning.py   # → qlearning_results.png
python ch04_dqn/dqn.py               # → dqn_results.png
python ch05_policy_gradient/reinforce.py  # → pg_results.png
```

No external dependencies beyond NumPy and Matplotlib.
