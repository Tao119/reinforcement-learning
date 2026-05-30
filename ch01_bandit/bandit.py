"""
ch01 — Multi-Armed Bandit
Agents: Epsilon-Greedy, UCB1, Thompson Sampling
Metric: cumulative regret over 1000 steps
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import MultiArmedBandit


# ─────────────────────────────────────── agents ────────────────────────────────

class EpsilonGreedy:
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select(self) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def best_arm(self) -> int:
        return int(np.argmax(self.values))


class UCB1:
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        # Pull each arm at least once
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return int(untried[0])
        ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def best_arm(self) -> int:
        return int(np.argmax(self.values))


class ThompsonSampling:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)   # successes + 1
        self.beta  = np.ones(n_arms)   # failures  + 1

    def select(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        self.alpha[arm] += reward
        self.beta[arm]  += 1.0 - reward

    def best_arm(self) -> int:
        return int(np.argmax(self.alpha / (self.alpha + self.beta)))


# ─────────────────────────────────────── run ───────────────────────────────────

def run_agent(agent, bandit: MultiArmedBandit, n_steps: int = 1000):
    regrets = []
    total_reward = 0.0
    cumulative_regret = 0.0
    for _ in range(n_steps):
        arm = agent.select()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        total_reward += reward
        cumulative_regret += bandit.best_prob - bandit.expected_reward(arm)
        regrets.append(cumulative_regret)
    return np.array(regrets), total_reward


def main():
    np.random.seed(0)
    n_steps = 1000
    bandit = MultiArmedBandit(n_arms=10, seed=42)

    print(f"Bandit arm probabilities: {np.round(bandit.probs, 3)}")
    print(f"True best arm: {bandit.best_arm} (prob={bandit.best_prob:.3f})\n")

    agents = {
        "Epsilon-Greedy (ε=0.1)": EpsilonGreedy(10, epsilon=0.1),
        "UCB1":                   UCB1(10, c=2.0),
        "Thompson Sampling":      ThompsonSampling(10),
    }

    results = {}
    for name, agent in agents.items():
        regrets, total_reward = run_agent(agent, bandit, n_steps)
        results[name] = (regrets, total_reward, agent.best_arm())
        print(f"[{name}]")
        print(f"  Converged to arm: {agent.best_arm()}")
        print(f"  Total reward:     {total_reward:.1f}")
        print(f"  Final regret:     {regrets[-1]:.2f}\n")

    # ── plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (regrets, _, _) in results.items():
        ax.plot(regrets, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Multi-Armed Bandit — Cumulative Regret")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "bandit_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    main()
