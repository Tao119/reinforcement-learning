"""
ch04 — DQN with pure NumPy
Architecture: Affine(16→32)→ReLU→Affine(32→4)
Features: experience replay, target network
"""
import sys
import os
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld

# ─────────────────────────── neural network (pure NumPy) ─────────────────────


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class Linear:
    def __init__(self, in_dim: int, out_dim: int, scale: float = 0.1):
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class QNetwork:
    """Two-layer MLP: state_dim → 32 → n_actions."""

    def __init__(self, state_dim: int, n_actions: int):
        self.fc1 = Linear(state_dim, 32)
        self.fc2 = Linear(32, n_actions)
        self._h: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = relu(self.fc1.forward(x))
        self._h = h
        return self.fc2.forward(h)

    def backward(self, dout: np.ndarray):
        dh = self.fc2.backward(dout)
        dh = dh * relu_grad(self.fc1._x @ self.fc1.W + self.fc1.b)
        self.fc1.backward(dh)

    def all_params(self):
        return self.fc1.params() + self.fc2.params()

    def copy_weights_from(self, other: "QNetwork"):
        self.fc1.W = other.fc1.W.copy()
        self.fc1.b = other.fc1.b.copy()
        self.fc2.W = other.fc2.W.copy()
        self.fc2.b = other.fc2.b.copy()


# ─────────────────────────── SGD optimizer ───────────────────────────────────

def sgd_update(params, lr: float = 0.01):
    for W, dW in params:
        W -= lr * dW


# ─────────────────────────── one-hot encoding ────────────────────────────────

def one_hot(state: int, n_states: int) -> np.ndarray:
    v = np.zeros((1, n_states))
    v[0, state] = 1.0
    return v


# ─────────────────────────── DQN agent ───────────────────────────────────────

class DQNAgent:
    def __init__(self, n_states: int, n_actions: int,
                 gamma: float = 0.99,
                 lr: float = 0.005,
                 buffer_capacity: int = 1000,
                 batch_size: int = 32,
                 target_update_freq: int = 100):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_net = QNetwork(n_states, n_actions)
        self.target_net = QNetwork(n_states, n_actions)
        self.target_net.copy_weights_from(self.q_net)

        self.buffer: deque = deque(maxlen=buffer_capacity)
        self.step_count = 0

    def select_action(self, state: int, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_net.forward(one_hot(state, self.n_states))
        return int(np.argmax(q))

    def store(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Build batch matrices
        S  = np.vstack([one_hot(s, self.n_states) for s in states])
        S_ = np.vstack([one_hot(s, self.n_states) for s in next_states])

        Q_current = self.q_net.forward(S)                # (B, n_actions)
        Q_target_vals = self.target_net.forward(S_)      # (B, n_actions)

        targets = Q_current.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(Q_target_vals[i])

        # MSE loss gradient: dL/dQ = 2*(Q - target) / B
        dout = 2.0 * (Q_current - targets) / self.batch_size
        self.q_net.backward(dout)
        sgd_update(self.q_net.all_params(), lr=self.lr)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.q_net)


# ─────────────────────────── training loop ───────────────────────────────────

def epsilon_schedule(ep, n_ep, start=1.0, end=0.01):
    return end + (start - end) * max(0.0, 1.0 - ep / n_ep)


def train_dqn(env: GridWorld, n_episodes: int = 3000) -> list:
    agent = DQNAgent(env.n_states, env.n_actions)
    rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        eps = epsilon_schedule(ep, n_episodes)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, eps)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            total_reward += reward
            state = next_state

        rewards.append(total_reward)

    return rewards


def train_qlearning_baseline(env: GridWorld, n_episodes: int = 3000,
                              alpha: float = 0.1, gamma: float = 0.99) -> list:
    Q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    for ep in range(n_episodes):
        state = env.reset()
        eps = epsilon_schedule(ep, n_episodes)
        total_reward = 0.0
        done = False
        while not done:
            if np.random.random() < eps:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(Q[state]))
            next_state, reward, done = env.step(action)
            best_next = np.max(Q[next_state]) if not done else 0.0
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return rewards


def moving_average(data, w=100):
    return np.convolve(data, np.ones(w) / w, mode="valid")


def main():
    np.random.seed(42)
    env = GridWorld()
    n_episodes = 3000

    print("=== DQN (pure NumPy) on 4×4 GridWorld ===")
    print("Architecture: Affine(16→32)→ReLU→Affine(32→4)")
    print(f"Episodes={n_episodes}, replay_buffer=1000, target_update=100 steps\n")

    print("Training DQN …")
    dqn_rewards = train_dqn(env, n_episodes)

    print("Training Q-Learning baseline …")
    ql_rewards  = train_qlearning_baseline(env, n_episodes)

    w = 100
    dqn_ma = moving_average(dqn_rewards, w)
    ql_ma  = moving_average(ql_rewards, w)

    print(f"\nDQN   avg reward (last {w} eps): {np.mean(dqn_rewards[-w:]):.3f}")
    print(f"QL    avg reward (last {w} eps): {np.mean(ql_rewards[-w:]):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(dqn_rewards, alpha=0.2, color="tomato")
    ax.plot(range(w-1, n_episodes), dqn_ma, color="tomato", lw=2, label="DQN")
    ax.plot(ql_rewards, alpha=0.2, color="steelblue")
    ax.plot(range(w-1, n_episodes), ql_ma, color="steelblue", lw=2, label="Q-Learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN vs Q-Learning — Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(range(w-1, n_episodes), dqn_ma, color="tomato",    lw=2, label="DQN")
    ax.plot(range(w-1, n_episodes), ql_ma,  color="steelblue", lw=2, label="Q-Learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Moving Avg Reward (100 eps)")
    ax.set_title("Convergence Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "dqn_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
