"""
ch09 — Advanced Multi-Armed Bandit Algorithms

Algorithms
----------
1. UCB1          — Upper Confidence Bound (Auer et al., 2002)
2. UCB-V         — Variance-aware UCB (Audibert et al., 2009)
3. KL-UCB        — KL-divergence confidence bound (Cappé et al., 2013)
4. EXP3          — Adversarial bandit (Auer et al., 2002)
5. LinUCB        — Contextual Linear UCB (Li et al., 2010)

Test environments
-----------------
- Stationary 10-arm Gaussian bandit (σ=1, μ ~ Uniform[-2, 2])
- Non-stationary bandit (means drift every 1000 steps)
- Bernoulli bandit (best suited for KL-UCB)
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════ Bandits ════════════════════════════════

class GaussianBandit:
    """k-arm Gaussian bandit, σ=1."""
    def __init__(self, n_arms: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_arms = n_arms
        self.means = rng.uniform(-2, 2, size=n_arms)
        self.best_arm = int(np.argmax(self.means))
        self.best_mean = float(self.means[self.best_arm])

    def pull(self, arm: int) -> float:
        return float(self.means[arm] + np.random.randn())

    def expected_reward(self, arm: int) -> float:
        return float(self.means[arm])

    def optimal_reward(self) -> float:
        return self.best_mean


class NonStationaryBandit:
    """Means drift by ±0.1 every `drift_every` steps."""
    def __init__(self, n_arms: int = 10, drift_every: int = 1000, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_arms = n_arms
        self.drift_every = drift_every
        self.means = rng.uniform(-2, 2, size=n_arms)
        self._step = 0
        self._rng = rng

    def pull(self, arm: int) -> float:
        self._step += 1
        if self._step % self.drift_every == 0:
            self.means += self._rng.uniform(-0.5, 0.5, size=self.n_arms)
        return float(self.means[arm] + np.random.randn())

    def expected_reward(self, arm: int) -> float:
        return float(self.means[arm])

    def optimal_reward(self) -> float:
        return float(self.means.max())


class BernoulliBandit:
    """k-arm Bernoulli bandit."""
    def __init__(self, n_arms: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_arms = n_arms
        self.probs = rng.uniform(0.1, 0.9, size=n_arms)
        self.best_arm = int(np.argmax(self.probs))
        self.best_prob = float(self.probs[self.best_arm])

    def pull(self, arm: int) -> float:
        return float(np.random.random() < self.probs[arm])

    def expected_reward(self, arm: int) -> float:
        return float(self.probs[arm])

    def optimal_reward(self) -> float:
        return self.best_prob


# ═══════════════════════════════════════ Agents ═════════════════════════════════

class UCB1Agent:
    """UCB1: a_t = argmax[μ̂_i + √(2 ln t / n_i)]."""
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        untried = np.where(self.counts == 0)[0]
        if len(untried):
            return int(untried[0])
        ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class UCBVAgent:
    """
    UCB-V: adds an empirical variance term.
    a_t = argmax[μ̂_i + √(2 V̂_i ln t / n_i) + 3b ln t / n_i]
    """
    def __init__(self, n_arms: int, b: float = 1.0, c: float = 1.0):
        self.n_arms = n_arms
        self.b = b
        self.c = c
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)
        self.sq_means = np.zeros(n_arms)   # E[X^2] for variance
        self.t = 0

    def select(self) -> int:
        self.t += 1
        untried = np.where(self.counts == 0)[0]
        if len(untried):
            return int(untried[0])
        variances = np.maximum(0.0, self.sq_means - self.means ** 2)
        ln_t = np.log(self.t)
        exploration = (np.sqrt(2 * variances * ln_t / self.counts)
                       + self.c * self.b * ln_t / self.counts)
        ucbv = self.means + exploration
        return int(np.argmax(ucbv))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n
        self.sq_means[arm] += (reward ** 2 - self.sq_means[arm]) / n


class KLUCBAgent:
    """
    KL-UCB for Bernoulli rewards.
    Solves: max q s.t. n_i * KL(μ̂_i, q) ≤ ln t + c ln ln t
    KL(p, q) = p ln(p/q) + (1-p) ln((1-p)/(1-q))
    Uses binary search.
    """
    def __init__(self, n_arms: int, c: float = 0.0, eps: float = 1e-6):
        self.n_arms = n_arms
        self.c = c
        self.eps = eps
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0

    @staticmethod
    def _kl_bernoulli(p: float, q: float, eps: float = 1e-10) -> float:
        p = np.clip(p, eps, 1 - eps)
        q = np.clip(q, eps, 1 - eps)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def _klucb_upper(self, mu: float, n: float, log_t: float) -> float:
        """Binary search for upper confidence bound."""
        bound = (log_t + self.c * np.log(max(log_t, 1))) / n
        lo, hi = mu, 1.0 - self.eps
        for _ in range(50):
            mid = (lo + hi) / 2
            if self._kl_bernoulli(mu, mid) < bound:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def select(self) -> int:
        self.t += 1
        untried = np.where(self.counts == 0)[0]
        if len(untried):
            return int(untried[0])
        log_t = np.log(self.t)
        ucb = np.array([
            self._klucb_upper(self.values[i], self.counts[i], log_t)
            for i in range(self.n_arms)
        ])
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class EXP3Agent:
    """
    EXP3 — Exponential-weight algorithm for Exploration and Exploitation.
    Designed for adversarial settings.  γ ∈ (0,1] is the mix parameter.
    """
    def __init__(self, n_arms: int, gamma: float = 0.1):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)

    @property
    def _probs(self) -> np.ndarray:
        w_sum = self.weights.sum()
        probs = (1 - self.gamma) * self.weights / w_sum + self.gamma / self.n_arms
        return probs

    def select(self) -> int:
        probs = self._probs
        return int(np.random.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float):
        probs = self._probs
        # Importance-weight estimate
        estimated_reward = reward / probs[arm]
        self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)
        # Clip to avoid overflow
        self.weights = np.clip(self.weights, 0, 1e30)


class LinUCBAgent:
    """
    LinUCB — Contextual bandit with linear reward model.
    Assumes E[r | a, x] = θ_a^T x.
    α controls exploration.
    """
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        # A_a = I + X_a^T X_a,  b_a = sum rewards * x
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self._last_context: np.ndarray | None = None

    def select(self, context: np.ndarray) -> int:
        self._last_context = context
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p[a] = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
        return int(np.argmax(p))

    def update(self, arm: int, reward: float):
        x = self._last_context
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


# ═══════════════════════════════════════ Runners ════════════════════════════════

def run_bandit(agent, bandit, n_steps: int, is_linucb: bool = False,
               context_dim: int = 5) -> tuple[np.ndarray, float]:
    """Run agent on bandit for n_steps; return (cumulative_regret, final_regret)."""
    rng = np.random.default_rng(99)
    regrets = []
    cum_regret = 0.0
    for _ in range(n_steps):
        if is_linucb:
            ctx = rng.standard_normal(context_dim)
            arm = agent.select(ctx)
        else:
            arm = agent.select()
        reward = bandit.pull(arm)
        if is_linucb:
            agent.update(arm, reward)
        else:
            agent.update(arm, reward)
        cum_regret += bandit.optimal_reward() - bandit.expected_reward(arm)
        regrets.append(cum_regret)
    return np.array(regrets), cum_regret


# ═══════════════════════════════════════ Main ═══════════════════════════════════

def main():
    n_steps = 5000
    n_arms = 10
    out_dir = os.path.dirname(__file__)
    results_all: dict[str, dict] = {}

    print("=" * 65)
    print("Advanced Multi-Armed Bandit Comparison")
    print("=" * 65)

    # ─── 1. Stationary Gaussian ─────────────────────────────────────────
    print("\n[1] Stationary 10-arm Gaussian Bandit")
    gaussian_bandit = GaussianBandit(n_arms=n_arms, seed=42)
    agents_gaussian = {
        "UCB1":  UCB1Agent(n_arms, c=2.0),
        "UCB-V": UCBVAgent(n_arms, b=1.0),
        "EXP3":  EXP3Agent(n_arms, gamma=0.1),
        "LinUCB (ctx=5)": LinUCBAgent(n_arms, context_dim=5, alpha=1.0),
    }
    results_gaussian: dict[str, np.ndarray] = {}
    for name, agent in agents_gaussian.items():
        is_lin = isinstance(agent, LinUCBAgent)
        regrets, final = run_bandit(agent, gaussian_bandit, n_steps,
                                    is_linucb=is_lin, context_dim=5)
        results_gaussian[name] = regrets
        results_all[f"gaussian_{name}"] = {"final_regret": round(final, 2)}
        print(f"  {name:22s}  final regret = {final:.2f}")

    # ─── 2. Non-stationary ──────────────────────────────────────────────
    print("\n[2] Non-Stationary Bandit (drift every 1000 steps)")
    ns_bandit = NonStationaryBandit(n_arms=n_arms, drift_every=1000, seed=42)
    agents_ns = {
        "UCB1":  UCB1Agent(n_arms, c=2.0),
        "UCB-V": UCBVAgent(n_arms, b=1.0),
        "EXP3":  EXP3Agent(n_arms, gamma=0.2),  # higher gamma for non-stationary
    }
    results_ns: dict[str, np.ndarray] = {}
    for name, agent in agents_ns.items():
        regrets, final = run_bandit(agent, ns_bandit, n_steps)
        results_ns[name] = regrets
        results_all[f"nonstationary_{name}"] = {"final_regret": round(final, 2)}
        print(f"  {name:22s}  final regret = {final:.2f}")

    # ─── 3. Bernoulli (KL-UCB shines here) ──────────────────────────────
    print("\n[3] Bernoulli Bandit (best for KL-UCB)")
    bern_bandit = BernoulliBandit(n_arms=n_arms, seed=42)
    agents_bern = {
        "UCB1":   UCB1Agent(n_arms, c=2.0),
        "KL-UCB": KLUCBAgent(n_arms),
        "EXP3":   EXP3Agent(n_arms, gamma=0.1),
    }
    results_bern: dict[str, np.ndarray] = {}
    for name, agent in agents_bern.items():
        regrets, final = run_bandit(agent, bern_bandit, n_steps)
        results_bern[name] = regrets
        results_all[f"bernoulli_{name}"] = {"final_regret": round(final, 2)}
        print(f"  {name:22s}  final regret = {final:.2f}")

    # ─── Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["steelblue", "darkorange", "forestgreen", "crimson", "purple"]

    for ax, results, title in [
        (axes[0], results_gaussian, "Gaussian Bandit"),
        (axes[1], results_ns,       "Non-Stationary Bandit"),
        (axes[2], results_bern,     "Bernoulli Bandit"),
    ]:
        for j, (name, regrets) in enumerate(results.items()):
            ax.plot(regrets, label=name, color=colors[j % len(colors)])
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Regret")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.suptitle("Advanced Multi-Armed Bandit — Regret Curves", fontsize=13)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "advanced_bandit_regret.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {png_path}")

    json_path = os.path.join(out_dir, "advanced_bandit_results.json")
    with open(json_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"Results saved → {json_path}")


if __name__ == "__main__":
    main()
