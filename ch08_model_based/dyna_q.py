"""
ch08 — Dyna-Q (model-based RL), pure NumPy
Extends Q-learning with a learned world model.
After each real step: n=10 simulated steps from stored (s,a)→(s',r) model.
Compares convergence speed vs plain Q-learning in real-world steps.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld


# ─────────────────────────── Dyna-Q ──────────────────────────────────────────

def epsilon_schedule(episode: int, n_episodes: int,
                     eps_start: float = 1.0, eps_end: float = 0.05) -> float:
    return eps_end + (eps_start - eps_end) * max(0.0, 1.0 - episode / n_episodes)


class DynaQ:
    """
    Q-learning augmented with a tabular world model.
    model[(s, a)] = (s', r)  — deterministic one-step model.
    """

    def __init__(self, env: GridWorld, alpha: float = 0.1,
                 gamma: float = 0.99, n_sim: int = 10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.n_sim = n_sim
        self.Q = np.zeros((env.n_states, env.n_actions))
        self.model: dict = {}          # (s, a) → (s', r)
        self.visited: list = []        # list of observed (s, a) pairs

    def select_action(self, state: int, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.env.n_actions)
        return int(np.argmax(self.Q[state]))

    def real_step(self, state: int, epsilon: float):
        action = self.select_action(state, epsilon)
        next_state, reward, done = self.env.step(action)

        # Q-update from real experience
        best_next = np.max(self.Q[next_state]) if not done else 0.0
        td_error = reward + self.gamma * best_next - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        # Model update
        key = (state, action)
        if key not in self.model:
            self.visited.append(key)
        self.model[key] = (next_state, reward)

        return next_state, reward, done

    def simulated_steps(self):
        if not self.visited:
            return
        for _ in range(self.n_sim):
            idx = np.random.randint(len(self.visited))
            s, a = self.visited[idx]
            s_prime, r = self.model[(s, a)]
            best_next = np.max(self.Q[s_prime])
            td_error = r + self.gamma * best_next - self.Q[s, a]
            self.Q[s, a] += self.alpha * td_error

    def train_episode(self, epsilon: float) -> tuple:
        state = self.env.reset()
        total_reward = 0.0
        real_steps = 0
        done = False
        max_steps = 300

        for _ in range(max_steps):
            next_state, reward, done = self.real_step(state, epsilon)
            total_reward += reward
            real_steps += 1
            self.simulated_steps()
            state = next_state
            if done:
                break

        return total_reward, real_steps


def train_dyna_q(env: GridWorld, n_episodes: int = 500,
                 n_sim: int = 10, alpha: float = 0.1,
                 gamma: float = 0.99) -> tuple:
    agent = DynaQ(env, alpha=alpha, gamma=gamma, n_sim=n_sim)
    episode_rewards = []
    cumulative_real_steps = []
    total_steps = 0

    for ep in range(n_episodes):
        epsilon = epsilon_schedule(ep, n_episodes)
        reward, steps = agent.train_episode(epsilon)
        episode_rewards.append(reward)
        total_steps += steps
        cumulative_real_steps.append(total_steps)

    return agent.Q, episode_rewards, cumulative_real_steps


def train_qlearning(env: GridWorld, n_episodes: int = 500,
                    alpha: float = 0.1, gamma: float = 0.99) -> tuple:
    Q = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []
    cumulative_real_steps = []
    total_steps = 0

    for ep in range(n_episodes):
        state = env.reset()
        epsilon = epsilon_schedule(ep, n_episodes)
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 300:
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            best_next = np.max(Q[next_state]) if not done else 0.0
            td_error = reward + gamma * best_next - Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state

        episode_rewards.append(total_reward)
        total_steps += steps
        cumulative_real_steps.append(total_steps)

    return Q, episode_rewards, cumulative_real_steps


def moving_average(data, w: int = 20) -> np.ndarray:
    return np.convolve(data, np.ones(w) / w, mode="valid")


def main():
    np.random.seed(42)
    env = GridWorld()
    n_episodes = 500

    print("=== Dyna-Q vs Q-Learning on 4×4 GridWorld ===")
    print(f"Episodes={n_episodes}, n_sim=10, alpha=0.1, gamma=0.99\n")

    # Q-learning
    np.random.seed(42)
    _, ql_rewards, ql_steps = train_qlearning(env, n_episodes=n_episodes)

    # Dyna-Q
    np.random.seed(42)
    _, dq_rewards, dq_steps = train_dyna_q(env, n_episodes=n_episodes, n_sim=10)

    w = 20
    ql_ma = moving_average(ql_rewards, w)
    dq_ma = moving_average(dq_rewards, w)

    print(f"Q-Learning  avg reward (last {w}): {np.mean(ql_rewards[-w:]):.3f}")
    print(f"Dyna-Q      avg reward (last {w}): {np.mean(dq_rewards[-w:]):.3f}")
    print(f"Q-Learning  total real steps: {ql_steps[-1]}")
    print(f"Dyna-Q      total real steps: {dq_steps[-1]}")

    # Find episode where running mean first exceeds 0.5 (rough convergence point)
    def first_solved(rewards, threshold: float = 0.5, window: int = 20) -> int:
        ma = moving_average(rewards, window)
        hits = np.where(ma >= threshold)[0]
        return int(hits[0]) + window if len(hits) > 0 else len(rewards)

    ql_ep = first_solved(ql_rewards)
    dq_ep = first_solved(dq_rewards)

    print(f"\nConvergence episode (avg>0.5 over {w}): "
          f"Q-Learning={ql_ep}  Dyna-Q={dq_ep}")
    print(f"Real steps at convergence: "
          f"Q-Learning={ql_steps[min(ql_ep, n_episodes-1)]}  "
          f"Dyna-Q={dq_steps[min(dq_ep, n_episodes-1)]}")

    # ── plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: reward vs episode
    ax = axes[0]
    ep_axis = range(len(ql_rewards))
    ax.plot(ql_rewards, alpha=0.15, color="steelblue")
    ax.plot(range(w - 1, n_episodes), ql_ma, color="steelblue", lw=2, label="Q-Learning")
    ax.plot(dq_rewards, alpha=0.15, color="darkorange")
    ax.plot(range(w - 1, n_episodes), dq_ma, color="darkorange", lw=2, label="Dyna-Q (n=10)")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="convergence threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Dyna-Q vs Q-Learning — Episode Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: reward vs real-world steps
    ax = axes[1]
    ax.plot(ql_steps, ql_rewards, alpha=0.15, color="steelblue")
    ax.plot(np.array(ql_steps)[w - 1:], ql_ma, color="steelblue", lw=2,
            label="Q-Learning")
    ax.plot(dq_steps, dq_rewards, alpha=0.15, color="darkorange")
    ax.plot(np.array(dq_steps)[w - 1:], dq_ma, color="darkorange", lw=2,
            label="Dyna-Q (n=10)")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="convergence threshold")
    ax.set_xlabel("Cumulative Real-World Steps")
    ax.set_ylabel("Total Reward")
    ax.set_title("Sample Efficiency — Real Steps to Convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "dynaq_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
