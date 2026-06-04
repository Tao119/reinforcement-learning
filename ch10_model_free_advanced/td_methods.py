"""
ch10 — Advanced Temporal-Difference Methods

Algorithms
----------
1. Q-Learning     — off-policy TD(0)
2. SARSA          — on-policy TD(0)
3. Expected SARSA — E[Q(s',a')] instead of max
4. Double Q-Learning — two Q-tables to reduce maximization bias
5. n-step TD      — n ∈ {1, 3, 5, 10} Q-learning variant

All evaluated on GridWorld (4×4) from common/env.py.
Metrics: average episode reward, convergence speed, final policy quality.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld

OUT_DIR = os.path.dirname(__file__)


# ═══════════════════════════════════════ Policy helpers ═════════════════════════

def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float,
                   n_actions: int) -> int:
    if np.random.random() < epsilon:
        return int(np.random.randint(n_actions))
    return int(np.argmax(Q[state]))


def greedy(Q: np.ndarray, state: int) -> int:
    return int(np.argmax(Q[state]))


# ═══════════════════════════════════════ Q-Learning ═════════════════════════════

def q_learning(env: GridWorld, n_episodes: int = 500,
               alpha: float = 0.1, gamma: float = 0.99,
               epsilon: float = 0.1) -> tuple[np.ndarray, list[float]]:
    Q = np.zeros((env.n_states, env.n_actions))
    ep_returns = []

    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        for _ in range(200):
            a = epsilon_greedy(Q, s, epsilon, env.n_actions)
            s2, r, done = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            s = s2
            total += r
            if done:
                break
        ep_returns.append(total)

    return Q, ep_returns


# ═══════════════════════════════════════ SARSA ══════════════════════════════════

def sarsa(env: GridWorld, n_episodes: int = 500,
          alpha: float = 0.1, gamma: float = 0.99,
          epsilon: float = 0.1) -> tuple[np.ndarray, list[float]]:
    Q = np.zeros((env.n_states, env.n_actions))
    ep_returns = []

    for _ in range(n_episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon, env.n_actions)
        total = 0.0
        for _ in range(200):
            s2, r, done = env.step(a)
            a2 = epsilon_greedy(Q, s2, epsilon, env.n_actions)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s, a = s2, a2
            total += r
            if done:
                break
        ep_returns.append(total)

    return Q, ep_returns


# ═══════════════════════════════════════ Expected SARSA ═════════════════════════

def expected_sarsa(env: GridWorld, n_episodes: int = 500,
                   alpha: float = 0.1, gamma: float = 0.99,
                   epsilon: float = 0.1) -> tuple[np.ndarray, list[float]]:
    Q = np.zeros((env.n_states, env.n_actions))
    ep_returns = []
    n_a = env.n_actions

    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        for _ in range(200):
            a = epsilon_greedy(Q, s, epsilon, n_a)
            s2, r, done = env.step(a)

            # Expected value under ε-greedy policy
            best_a2 = int(np.argmax(Q[s2]))
            probs = np.full(n_a, epsilon / n_a)
            probs[best_a2] += 1 - epsilon
            expected_q = float(probs @ Q[s2])

            Q[s, a] += alpha * (r + gamma * expected_q - Q[s, a])
            s = s2
            total += r
            if done:
                break
        ep_returns.append(total)

    return Q, ep_returns


# ═══════════════════════════════════════ Double Q-Learning ══════════════════════

def double_q_learning(env: GridWorld, n_episodes: int = 500,
                      alpha: float = 0.1, gamma: float = 0.99,
                      epsilon: float = 0.1) -> tuple[np.ndarray, list[float]]:
    QA = np.zeros((env.n_states, env.n_actions))
    QB = np.zeros((env.n_states, env.n_actions))
    ep_returns = []

    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        for _ in range(200):
            # Select action using combined Q
            combined = QA + QB
            a = epsilon_greedy(combined, s, epsilon, env.n_actions)
            s2, r, done = env.step(a)

            if np.random.random() < 0.5:
                # Update QA using QB for evaluation
                best_a2 = int(np.argmax(QA[s2]))
                QA[s, a] += alpha * (r + gamma * QB[s2, best_a2] - QA[s, a])
            else:
                # Update QB using QA for evaluation
                best_a2 = int(np.argmax(QB[s2]))
                QB[s, a] += alpha * (r + gamma * QA[s2, best_a2] - QB[s, a])

            s = s2
            total += r
            if done:
                break
        ep_returns.append(total)

    Q = (QA + QB) / 2
    return Q, ep_returns


# ═══════════════════════════════════════ n-step TD ══════════════════════════════

def n_step_q_learning(env: GridWorld, n: int = 3, n_episodes: int = 500,
                      alpha: float = 0.1, gamma: float = 0.99,
                      epsilon: float = 0.1) -> tuple[np.ndarray, list[float]]:
    """n-step Q-learning (semi-gradient, off-policy via n-step returns)."""
    Q = np.zeros((env.n_states, env.n_actions))
    ep_returns = []
    gammas = np.array([gamma ** i for i in range(n + 1)])

    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        # Store transitions
        states = [s]
        actions: list[int] = []
        rewards: list[float] = []
        done_flags: list[bool] = []

        for t in range(200 + n):
            if not done_flags or not done_flags[-1]:
                a = epsilon_greedy(Q, states[-1], epsilon, env.n_actions)
                s2, r, done = env.step(a)
                actions.append(a)
                rewards.append(r)
                states.append(s2)
                done_flags.append(done)
                total += r

            tau = t - n + 1
            if tau >= 0:
                # n-step return
                t_end = min(tau + n, len(rewards))
                G = sum(gammas[i - tau] * rewards[i] for i in range(tau, t_end))
                if t_end < len(rewards) and not done_flags[t_end - 1]:
                    G += gammas[n] * Q[states[t_end]].max()
                st, at = states[tau], actions[tau]
                Q[st, at] += alpha * (G - Q[st, at])

            if len(done_flags) > 0 and done_flags[-1] and tau >= len(rewards) - 1:
                break

        ep_returns.append(total)

    return Q, ep_returns


# ═══════════════════════════════════════ Smoothing ══════════════════════════════

def _smooth(x: list[float], k: int = 20) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if len(arr) < k:
        return arr
    kernel = np.ones(k) / k
    return np.convolve(arr, kernel, mode="valid")


# ═══════════════════════════════════════ Main ═══════════════════════════════════

def main():
    np.random.seed(42)
    env = GridWorld()
    n_episodes = 600
    results: dict[str, dict] = {}

    print("=" * 60)
    print("Advanced TD Methods on GridWorld (4×4)")
    print("=" * 60)

    # ─── Core algorithms ────────────────────────────────────────────────
    algorithms = {
        "Q-Learning":       q_learning,
        "SARSA":            sarsa,
        "Expected SARSA":   expected_sarsa,
        "Double Q-Learning": double_q_learning,
    }
    ep_curves: dict[str, list[float]] = {}
    for name, fn in algorithms.items():
        Q, returns = fn(env, n_episodes=n_episodes)
        ep_curves[name] = returns
        avg_last100 = float(np.mean(returns[-100:]))
        results[name] = {"avg_last100_return": round(avg_last100, 4)}
        print(f"  {name:22s}  avg(last 100 ep) = {avg_last100:.4f}")

    # ─── n-step comparison ──────────────────────────────────────────────
    print("\n  n-step Q-Learning:")
    for n in [1, 3, 5, 10]:
        _, returns = n_step_q_learning(env, n=n, n_episodes=n_episodes)
        ep_curves[f"n-step (n={n})"] = returns
        avg = float(np.mean(returns[-100:]))
        results[f"nstep_n{n}"] = {"avg_last100_return": round(avg, 4)}
        print(f"    n={n:2d}  avg(last 100 ep) = {avg:.4f}")

    # ─── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    smooth_k = 20
    colors = ["steelblue", "darkorange", "forestgreen", "crimson",
              "purple", "brown", "pink", "gray"]

    # Left: main algorithms
    ax = axes[0]
    main_keys = list(algorithms.keys())
    for j, name in enumerate(main_keys):
        smoothed = _smooth(ep_curves[name], smooth_k)
        ax.plot(smoothed, label=name, color=colors[j % len(colors)])
    ax.set_title("Core TD Methods")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return (smoothed k={smooth_k})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Right: n-step
    ax2 = axes[1]
    n_colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
    for j, n in enumerate([1, 3, 5, 10]):
        key = f"n-step (n={n})"
        smoothed = _smooth(ep_curves[key], smooth_k)
        ax2.plot(smoothed, label=key, color=n_colors[j])
    ax2.set_title("n-step Q-Learning Comparison")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(f"Return (smoothed k={smooth_k})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    plt.suptitle("Advanced TD Methods — GridWorld", fontsize=13)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "td_methods_comparison.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {png}")

    json_path = os.path.join(OUT_DIR, "td_methods_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {json_path}")


if __name__ == "__main__":
    main()
