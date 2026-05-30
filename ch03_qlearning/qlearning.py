"""
ch03 — Tabular Q-Learning on GridWorld
Q(s,a) ← Q(s,a) + alpha*(r + gamma*max_a'Q(s',a') - Q(s,a))
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld


def epsilon_schedule(episode: int, n_episodes: int,
                     eps_start: float = 1.0, eps_end: float = 0.01) -> float:
    """Linear annealing from eps_start to eps_end."""
    return eps_end + (eps_start - eps_end) * max(0.0, 1.0 - episode / n_episodes)


def train(env: GridWorld, n_episodes: int = 5000,
          alpha: float = 0.1, gamma: float = 0.99) -> tuple:
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        epsilon = epsilon_schedule(ep, n_episodes)
        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-update
            best_next = np.max(Q[next_state]) if not done else 0.0
            td_error = reward + gamma * best_next - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

        episode_rewards.append(total_reward)

    return Q, episode_rewards


def extract_policy(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=1)


def print_q_table(env: GridWorld, Q: np.ndarray):
    symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    print("\n=== Final Q-Table (max Q per state) ===")
    for r in range(env.height):
        row_str = []
        for c in range(env.width):
            s = r * env.width + c
            row_str.append(f"{np.max(Q[s]):+6.3f}")
        print("  ".join(row_str))

    print("\n=== Greedy Policy ===")
    for r in range(env.height):
        row_str = []
        for c in range(env.width):
            s = r * env.width + c
            if (r, c) == env.goal:
                row_str.append(" G ")
            elif (r, c) == env.hole:
                row_str.append(" X ")
            else:
                row_str.append(f" {symbols[np.argmax(Q[s])]} ")
        print(" | ".join(row_str))


def moving_average(data, window: int = 100) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def main():
    np.random.seed(42)
    env = GridWorld()
    n_episodes = 5000

    print("=== Tabular Q-Learning on 4×4 GridWorld ===")
    print(f"Episodes={n_episodes}, alpha=0.1, gamma=0.99, ε annealed 1.0→0.01\n")

    Q, rewards = train(env, n_episodes=n_episodes)

    print_q_table(env, Q)

    smoothed = moving_average(rewards, window=100)
    print(f"\nAverage reward (last 100 episodes): {np.mean(rewards[-100:]):.3f}")

    # ── plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards, alpha=0.25, color="steelblue", label="episode reward")
    ax.plot(range(99, len(rewards)), smoothed, color="steelblue", lw=2, label="moving avg (100)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning — Episode Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "qlearning_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
