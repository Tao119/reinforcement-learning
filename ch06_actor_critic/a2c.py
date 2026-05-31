"""
ch06 — Advantage Actor-Critic (A2C), pure NumPy
Actor:  Affine(state_dim→32)→ReLU→Affine(32→n_actions)→Softmax
Critic: Affine(state_dim→32)→ReLU→Affine(32→1)  (state value function)
Update: online (every step), advantage = r + gamma*V(s') - V(s)
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld

# Also need Q-learning and REINFORCE for comparison
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch03_qlearning"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch05_policy_gradient"))
import qlearning
import reinforce


# ─────────────────────────── utility functions ───────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


# ─────────────────────────── Actor network ───────────────────────────────────

class ActorNetwork:
    """Two-layer softmax policy: state_dim → 32 → n_actions."""

    def __init__(self, state_dim: int, n_actions: int, lr: float = 0.005):
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / 32)
        self.W1 = scale1 * np.random.randn(state_dim, 32)
        self.b1 = np.zeros(32)
        self.W2 = scale2 * np.random.randn(32, n_actions)
        self.b2 = np.zeros(n_actions)
        self.lr = lr
        self.n_actions = n_actions
        self.state_dim = state_dim

    def _one_hot(self, state: int) -> np.ndarray:
        x = np.zeros(self.state_dim)
        x[state] = 1.0
        return x

    def forward(self, state: int):
        x = self._one_hot(state)
        h = relu(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = softmax(logits)
        return probs, x, h, logits

    def select_action(self, state: int) -> int:
        probs, _, _, _ = self.forward(state)
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, state: int, action: int, advantage: float):
        probs, x, h, _ = self.forward(state)
        # gradient of log pi(a|s): dlogits = probs, then zero out action and subtract 1
        dlogits = probs.copy()
        dlogits[action] -= 1.0
        # actor loss = -log_pi(a|s) * advantage → gradient ascent
        dlogits *= -advantage

        # backprop through W2
        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.W2.T

        # backprop through relu
        dh_pre = dh * relu_grad(x @ self.W1 + self.b1)

        # backprop through W1
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


# ─────────────────────────── Critic network ──────────────────────────────────

class CriticNetwork:
    """Two-layer value function: state_dim → 32 → 1."""

    def __init__(self, state_dim: int, lr: float = 0.01):
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / 32)
        self.W1 = scale1 * np.random.randn(state_dim, 32)
        self.b1 = np.zeros(32)
        self.W2 = scale2 * np.random.randn(32, 1)
        self.b2 = np.zeros(1)
        self.lr = lr
        self.state_dim = state_dim

    def _one_hot(self, state: int) -> np.ndarray:
        x = np.zeros(self.state_dim)
        x[state] = 1.0
        return x

    def predict(self, state: int) -> float:
        x = self._one_hot(state)
        h = relu(x @ self.W1 + self.b1)
        v = float((h @ self.W2 + self.b2).squeeze())
        return v

    def update(self, state: int, advantage: float):
        x = self._one_hot(state)
        h_pre = x @ self.W1 + self.b1
        h = relu(h_pre)
        # critic loss = advantage^2, gradient = 2*advantage (but advantage = td_error here)
        # ∂L/∂v = -2 * advantage  (since advantage = target - v, loss = advantage^2)
        dv = np.array([-2.0 * advantage])

        dW2 = np.outer(h, dv)
        db2 = dv
        dh = dv @ self.W2.T

        dh_pre = dh * relu_grad(h_pre)
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


# ─────────────────────────── A2C training ────────────────────────────────────

def train_a2c(env: GridWorld, n_episodes: int = 3000,
              gamma: float = 0.99,
              actor_lr: float = 0.001,
              critic_lr: float = 0.005,
              entropy_coef: float = 0.05,
              n_step: int = 5) -> list:
    """
    Train A2C with n-step returns and entropy regularisation.

    Uses n-step Monte-Carlo returns instead of 1-step TD to reduce the
    variance of advantage estimates.  Entropy regularisation (entropy_coef)
    encourages exploration and prevents the policy from collapsing onto
    suboptimal deterministic actions.

    Parameters
    ----------
    n_step : int
        Number of steps to accumulate before computing a bootstrapped return.
        Larger values lower variance at the cost of higher bias.
    entropy_coef : float
        Weight on the entropy bonus added to the actor loss gradient.
    """
    actor = ActorNetwork(env.n_states, env.n_actions, lr=actor_lr)
    critic = CriticNetwork(env.n_states, lr=critic_lr)
    episode_rewards = []

    for ep in range(n_episodes):
        # Linearly decay epsilon from 0.5 → 0 over first 20% of episodes.
        # Ensures the agent visits the goal state early, providing a positive
        # reward signal before the critic is well-calibrated.
        epsilon = max(0.0, 0.5 * (1.0 - ep / (0.2 * n_episodes)))

        state = env.reset()
        total_reward = 0.0
        done = False
        max_steps = 300

        # Buffers for n-step return computation
        buf_states: list = []
        buf_actions: list = []
        buf_rewards: list = []
        buf_probs: list = []

        def _flush_buffer(terminal_state, terminal_done):
            """Apply n-step updates for accumulated transitions."""
            if not buf_states:
                return
            # Bootstrap from last state
            v_boot = 0.0 if terminal_done else critic.predict(terminal_state)
            G = v_boot
            for i in reversed(range(len(buf_states))):
                G = buf_rewards[i] + gamma * G
                s = buf_states[i]
                a = buf_actions[i]
                probs = buf_probs[i]
                v_s = critic.predict(s)
                advantage = G - v_s

                # Actor gradient (policy gradient + entropy)
                dlogits = probs.copy()
                dlogits[a] -= 1.0
                dlogits *= -advantage
                entropy_grad = entropy_coef * (np.log(probs + 1e-8) + 1.0)
                dlogits += entropy_grad

                x = actor._one_hot(s)
                h_pre = x @ actor.W1 + actor.b1
                h = relu(h_pre)
                dW2 = np.outer(h, dlogits)
                db2 = dlogits
                dh = dlogits @ actor.W2.T
                dh_pre = dh * relu_grad(h_pre)
                dW1 = np.outer(x, dh_pre)
                db1 = dh_pre

                actor.W1 -= actor.lr * dW1
                actor.b1 -= actor.lr * db1
                actor.W2 -= actor.lr * dW2
                actor.b2 -= actor.lr * db2

                critic.update(s, advantage)

        for step in range(max_steps):
            probs, _, _, _ = actor.forward(state)
            if epsilon > 0 and np.random.random() < epsilon:
                action = int(np.random.randint(actor.n_actions))
            else:
                action = int(np.random.choice(actor.n_actions, p=probs))
            next_state, reward, done = env.step(action)
            total_reward += reward

            buf_states.append(state)
            buf_actions.append(action)
            buf_rewards.append(reward)
            buf_probs.append(probs)

            if len(buf_states) >= n_step or done:
                _flush_buffer(next_state, done)
                buf_states.clear()
                buf_actions.clear()
                buf_rewards.clear()
                buf_probs.clear()

            state = next_state
            if done:
                break

        # Flush any remaining transitions
        _flush_buffer(state, done)

        episode_rewards.append(total_reward)

    return episode_rewards


def moving_average(data, w: int = 100) -> np.ndarray:
    return np.convolve(data, np.ones(w) / w, mode="valid")


def main():
    np.random.seed(42)
    env = GridWorld()

    print("=== A2C: Advantage Actor-Critic on 4×4 GridWorld ===")
    print("Actor:  state→32→ReLU→n_actions→Softmax")
    print("Critic: state→32→ReLU→1  (V function)")
    print("Update: n-step returns (n=5), entropy bonus, epsilon-greedy warmup")
    print()
    print("Note: on small GridWorlds with sparse + terminal hole rewards,")
    print("tabular Q-learning outperforms neural A2C because exact Bellman")
    print("propagation avoids function-approximation error in the critic.\n")

    # ── A2C  (actor_lr=0.001, critic_lr=0.005, entropy_coef=0.01)
    n_a2c = 3000
    rewards_a2c = train_a2c(env, n_episodes=n_a2c,
                            actor_lr=0.001, critic_lr=0.005, entropy_coef=0.01)
    print(f"A2C avg reward (last 100): {np.mean(rewards_a2c[-100:]):.3f}")

    # ── Q-learning (3000 episodes for fair comparison)
    np.random.seed(42)
    _, rewards_ql = qlearning.train(env, n_episodes=n_a2c)
    print(f"Q-Learning avg reward (last 100): {np.mean(rewards_ql[-100:]):.3f}")

    # ── REINFORCE (3000 episodes)
    np.random.seed(42)
    _, rewards_pg = reinforce.train(env, n_episodes=n_a2c)
    print(f"REINFORCE avg reward (last 100): {np.mean(rewards_pg[-100:]):.3f}")

    # ── summary table
    w = 100
    results = {
        "Q-Learning":  (rewards_ql,  "steelblue"),
        "REINFORCE":   (rewards_pg,  "purple"),
        "A2C":         (rewards_a2c, "darkorange"),
    }

    print("\n=== Performance Summary (last 100 episodes) ===")
    print(f"{'Method':<14} {'Mean':>8} {'Std':>8} {'Max':>8}")
    print("-" * 42)
    for name, (r, _) in results.items():
        tail = r[-100:]
        print(f"{name:<14} {np.mean(tail):>8.3f} {np.std(tail):>8.3f} {np.max(tail):>8.3f}")

    # ── plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, (r, color) in results.items():
        ma = moving_average(r, w)
        ax.plot(r, alpha=0.12, color=color)
        ax.plot(range(w - 1, len(r)), ma, color=color, lw=2, label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning vs REINFORCE vs A2C\n(moving average, window=100)")
    ax.legend()
    ax.grid(alpha=0.3)

    # summary table as second panel
    ax = axes[1]
    ax.axis("off")
    col_labels = ["Method", "Mean (last 100)", "Std", "Max"]
    rows = []
    for name, (r, _) in results.items():
        tail = r[-100:]
        rows.append([name, f"{np.mean(tail):.3f}", f"{np.std(tail):.3f}", f"{np.max(tail):.3f}"])
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    ax.set_title("Performance Comparison", pad=20)

    out = os.path.join(os.path.dirname(__file__), "a2c_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
