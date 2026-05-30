"""
ch05 — REINFORCE (Policy Gradient) with baseline
Policy: softmax over Affine(state_dim→n_actions), pure NumPy
Update: θ ← θ + α * (G_t - b) * ∇log π(a|s)
Baseline: running average of episode returns
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld


# ──────────────────────────── softmax policy ─────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)   # numerical stability
    e = np.exp(x)
    return e / e.sum()


class PolicyNetwork:
    """Linear softmax policy: state (one-hot) → action probabilities."""

    def __init__(self, state_dim: int, n_actions: int, lr: float = 0.01):
        self.W = np.zeros((state_dim, n_actions))
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.lr = lr

    def forward(self, state: int) -> np.ndarray:
        x = np.zeros(self.state_dim)
        x[state] = 1.0
        logits = x @ self.W
        return softmax(logits)

    def select_action(self, state: int) -> int:
        probs = self.forward(state)
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, trajectory: list, baseline: float, gamma: float):
        """
        trajectory: list of (state, action, reward)
        baseline:   running average of returns (scalar)
        """
        T = len(trajectory)
        returns = np.zeros(T)

        # Compute discounted returns G_t
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][2] + gamma * G
            returns[t] = G

        # REINFORCE gradient ascent
        for t, (state, action, _) in enumerate(trajectory):
            x = np.zeros(self.state_dim)
            x[state] = 1.0
            probs = softmax(x @ self.W)

            # ∇log π(a|s) = x ⊗ (e_a - π(s))
            grad = np.outer(x, -probs)
            grad[:, action] += x

            advantage = returns[t] - baseline
            self.W += self.lr * advantage * grad


# ─────────────────────────── training loop ───────────────────────────────────

def train(env: GridWorld, n_episodes: int = 2000,
          gamma: float = 0.99, lr: float = 0.02) -> tuple:
    policy = PolicyNetwork(env.n_states, env.n_actions, lr=lr)
    episode_returns = []
    running_baseline = 0.0
    baseline_alpha = 0.05   # EMA coefficient for baseline

    for ep in range(n_episodes):
        state = env.reset()
        trajectory = []
        done = False
        max_steps = 200

        for _ in range(max_steps):
            action = policy.select_action(state)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                break

        # Compute episode return
        G = sum(r * gamma**t for t, (_, _, r) in enumerate(trajectory))
        episode_returns.append(G)

        # Update baseline (running EMA of returns)
        running_baseline += baseline_alpha * (G - running_baseline)

        # Policy update
        policy.update(trajectory, running_baseline, gamma)

    return policy, episode_returns


def moving_average(data, w=50):
    return np.convolve(data, np.ones(w) / w, mode="valid")


def main():
    np.random.seed(42)
    env = GridWorld()
    n_episodes = 2000

    print("=== REINFORCE (Policy Gradient) on 4×4 GridWorld ===")
    print("Policy: softmax(W @ one_hot(state))  — pure NumPy")
    print(f"Episodes={n_episodes}, γ=0.99, α=0.02, baseline=running EMA\n")

    policy, returns = train(env, n_episodes=n_episodes)

    w = 50
    ma = moving_average(returns, w)
    print(f"Avg return (last {w} episodes): {np.mean(returns[-w:]):.3f}")
    print(f"Avg return (first {w} episodes): {np.mean(returns[:w]):.3f}")

    print("\n=== Greedy Policy (from learned softmax) ===")
    symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    for r in range(env.height):
        row = []
        for c in range(env.width):
            s = r * env.width + c
            if (r, c) == env.goal:
                row.append(" G ")
            elif (r, c) == env.hole:
                row.append(" X ")
            else:
                probs = policy.forward(s)
                row.append(f" {symbols[int(np.argmax(probs))]} ")
        print(" | ".join(row))

    # ── plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(returns, alpha=0.25, color="purple", label="episode return")
    ax.plot(range(w-1, n_episodes), ma, color="purple", lw=2, label=f"moving avg ({w})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Discounted Return")
    ax.set_title("REINFORCE — Episode Returns")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.imshow(policy.W.T, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("State")
    ax.set_ylabel("Action")
    ax.set_yticks(range(env.n_actions))
    ax.set_yticklabels(["↑ up", "↓ down", "← left", "→ right"])
    ax.set_title("Learned Policy Weights W")
    ax.colorbar = fig.colorbar(ax.images[0], ax=ax)

    out = os.path.join(os.path.dirname(__file__), "pg_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
