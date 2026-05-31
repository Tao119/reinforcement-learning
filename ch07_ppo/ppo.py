"""
ch07 — Proximal Policy Optimization (PPO-Clip), pure NumPy
Same Actor-Critic structure as A2C.
Collect trajectories of length 64, then update K=4 epochs with clipped objective.
L_clip = min(r_t * A, clip(r_t, 1-eps, 1+eps) * A)
eps = 0.2
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld


# ─────────────────────────── utilities ───────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


# ─────────────────────────── networks ────────────────────────────────────────

class ActorCritic:
    """
    Shared backbone not used here — actor and critic have independent weights
    to keep the PPO implementation transparent.
    """

    def __init__(self, state_dim: int, n_actions: int,
                 actor_lr: float = 0.003, critic_lr: float = 0.01):
        s1 = np.sqrt(2.0 / state_dim)
        s2 = np.sqrt(2.0 / 32)

        # Actor weights
        self.aW1 = s1 * np.random.randn(state_dim, 32)
        self.ab1 = np.zeros(32)
        self.aW2 = s2 * np.random.randn(32, n_actions)
        self.ab2 = np.zeros(n_actions)

        # Critic weights
        self.cW1 = s1 * np.random.randn(state_dim, 32)
        self.cb1 = np.zeros(32)
        self.cW2 = s2 * np.random.randn(32, 1)
        self.cb2 = np.zeros(1)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.n_actions = n_actions

    def _one_hot(self, state: int) -> np.ndarray:
        x = np.zeros(self.state_dim)
        x[state] = 1.0
        return x

    def actor_forward(self, state: int):
        x = self._one_hot(state)
        h_pre = x @ self.aW1 + self.ab1
        h = relu(h_pre)
        logits = h @ self.aW2 + self.ab2
        probs = softmax(logits)
        return probs, x, h_pre, h

    def critic_forward(self, state: int) -> float:
        x = self._one_hot(state)
        h = relu(x @ self.cW1 + self.cb1)
        v = float((h @ self.cW2 + self.cb2).squeeze())
        return v

    def select_action(self, state: int):
        probs, _, _, _ = self.actor_forward(state)
        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob

    def actor_update_ppo(self, state: int, action: int, advantage: float,
                         old_log_prob: float, eps: float = 0.2) -> tuple:
        """Single-sample PPO-Clip actor update. Returns (clipped, clip_frac)."""
        probs, x, h_pre, h = self.actor_forward(state)
        log_prob = float(np.log(probs[action] + 1e-8))

        ratio = np.exp(log_prob - old_log_prob)
        clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps)
        was_clipped = int(ratio != clipped)

        # Use clipped objective: -min(r*A, clip(r,1-e,1+e)*A)
        if advantage >= 0:
            # both terms >= 0; take minimum (more conservative)
            use_clip = ratio > (1.0 + eps)
        else:
            # both terms <= 0; take minimum (less negative = less update)
            use_clip = ratio < (1.0 - eps)

        effective_ratio = clipped if use_clip else ratio
        # gradient of -effective_ratio * advantage w.r.t. log_prob
        # d(ratio)/d(log_prob) = ratio, but only when not clipped
        if use_clip:
            d_loss_d_logprob = 0.0   # gradient zeroed by clip
        else:
            d_loss_d_logprob = -ratio * advantage  # -d(r*A)/d(log_prob)

        # Chain rule: d(log_prob)/d(logits) = e_a - pi
        dlogits = probs.copy()
        dlogits[action] -= 1.0
        dlogits *= d_loss_d_logprob

        # Backprop W2
        dW2 = np.outer(h, dlogits)
        db2 = dlogits
        dh = dlogits @ self.aW2.T
        dh_pre = dh * relu_grad(h_pre)
        x = self._one_hot(state)
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        self.aW1 -= self.actor_lr * dW1
        self.ab1 -= self.actor_lr * db1
        self.aW2 -= self.actor_lr * dW2
        self.ab2 -= self.actor_lr * db2

        return effective_ratio * advantage, was_clipped

    def critic_update(self, state: int, target: float):
        x = self._one_hot(state)
        h_pre = x @ self.cW1 + self.cb1
        h = relu(h_pre)
        v = float((h @ self.cW2 + self.cb2).squeeze())

        dv = np.array([2.0 * (v - target)])
        dW2 = np.outer(h, dv)
        db2 = dv
        dh = dv @ self.cW2.T
        dh_pre = dh * relu_grad(h_pre)
        dW1 = np.outer(x, dh_pre)
        db1 = dh_pre

        self.cW1 -= self.critic_lr * dW1
        self.cb1 -= self.critic_lr * db1
        self.cW2 -= self.critic_lr * dW2
        self.cb2 -= self.critic_lr * db2


# ─────────────────────────── trajectory collection ───────────────────────────

def collect_trajectory(env: GridWorld, ac: ActorCritic,
                       n_steps: int = 64,
                       epsilon: float = 0.0) -> list:
    """
    Collect n_steps transitions, respecting episode boundaries.

    Parameters
    ----------
    epsilon : float
        Probability of taking a random action (epsilon-greedy exploration).
        Setting epsilon > 0 during early training helps the agent discover
        sparse positive rewards before the policy can exploit them.
    """
    traj = []
    state = env.reset()
    for _ in range(n_steps):
        if epsilon > 0 and np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
            probs, _, _, _ = ac.actor_forward(state)
            log_prob = float(np.log(probs[action] + 1e-8))
        else:
            action, log_prob = ac.select_action(state)
        next_state, reward, done = env.step(action)
        traj.append((state, action, reward, log_prob, done, next_state))
        if done:
            state = env.reset()
        else:
            state = next_state
    return traj


def compute_returns(traj: list, ac: ActorCritic,
                    gamma: float = 0.99) -> list:
    """Compute bootstrapped advantages for each transition."""
    transitions = []
    for state, action, reward, log_prob, done, next_state in traj:
        v_s = ac.critic_forward(state)
        v_next = 0.0 if done else ac.critic_forward(next_state)
        advantage = reward + gamma * v_next - v_s
        target = reward + gamma * v_next
        transitions.append((state, action, advantage, log_prob, target))
    return transitions


# ─────────────────────────── PPO training ────────────────────────────────────

def train_ppo(env: GridWorld, n_updates: int = 1000,
              traj_len: int = 64, n_epochs: int = 4,
              eps: float = 0.2, gamma: float = 0.99,
              actor_lr: float = 0.001, critic_lr: float = 0.005) -> dict:
    ac = ActorCritic(env.n_states, env.n_actions,
                     actor_lr=actor_lr, critic_lr=critic_lr)

    all_rewards = []
    clip_fracs = []
    entropies = []
    episode_rewards = []
    current_ep_reward = 0.0
    state = env.reset()

    for update in range(n_updates):
        # Linearly decay epsilon from 0.3 → 0 over the first 30% of updates
        epsilon = max(0.0, 0.3 * (1.0 - update / (0.3 * n_updates)))

        # Collect trajectory
        traj = collect_trajectory(env, ac, n_steps=traj_len, epsilon=epsilon)
        transitions = compute_returns(traj, ac, gamma=gamma)

        # Track episode rewards from trajectory
        for _, _, reward, _, done, _ in traj:
            current_ep_reward += reward
            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0

        # PPO epochs
        update_clip_count = 0
        total_transitions = len(transitions)

        for epoch in range(n_epochs):
            idx = np.random.permutation(len(transitions))
            for i in idx:
                s, a, adv, old_lp, target = transitions[i]
                _, was_clipped = ac.actor_update_ppo(s, a, adv, old_lp, eps)
                ac.critic_update(s, target)
                update_clip_count += was_clipped

        clip_frac = update_clip_count / (total_transitions * n_epochs)
        clip_fracs.append(clip_frac)

        # Policy entropy (averaged over states)
        ent = 0.0
        for s_idx in range(env.n_states):
            probs, _, _, _ = ac.actor_forward(s_idx)
            ent += -float(np.sum(probs * np.log(probs + 1e-8)))
        entropies.append(ent / env.n_states)

        if (update + 1) % 100 == 0:
            avg_rew = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            print(f"update {update+1:5d}: ep_reward={avg_rew:.3f}  "
                  f"clip_frac={clip_frac:.3f}  entropy={entropies[-1]:.3f}")

    return {
        "episode_rewards": episode_rewards,
        "clip_fracs": clip_fracs,
        "entropies": entropies,
    }


def moving_average(data, w: int = 20) -> np.ndarray:
    return np.convolve(data, np.ones(w) / w, mode="valid")


def main():
    np.random.seed(42)
    env = GridWorld()

    print("=== PPO-Clip on 4×4 GridWorld ===")
    print("Trajectory length=64, K=4 epochs, eps=0.2, gamma=0.99\n")

    results = train_ppo(env, n_updates=1000,
                        actor_lr=0.001, critic_lr=0.005)

    ep_rewards = results["episode_rewards"]
    clip_fracs = results["clip_fracs"]
    entropies = results["entropies"]

    print(f"\nTotal episodes: {len(ep_rewards)}")
    print(f"Avg reward (last 50 episodes): {np.mean(ep_rewards[-50:]):.3f}")
    print(f"Avg clip fraction: {np.mean(clip_fracs):.3f}")
    print(f"Final entropy: {entropies[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Episode rewards
    ax = axes[0]
    if len(ep_rewards) > 20:
        ma = moving_average(ep_rewards, 20)
        ax.plot(ep_rewards, alpha=0.2, color="teal", label="episode reward")
        ax.plot(range(19, len(ep_rewards)), ma, color="teal", lw=2,
                label="moving avg (20)")
    else:
        ax.plot(ep_rewards, color="teal", label="episode reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("PPO — Episode Rewards")
    ax.legend()
    ax.grid(alpha=0.3)

    # Clip fraction per update
    ax = axes[1]
    updates = range(1, len(clip_fracs) + 1)
    ax.plot(updates, clip_fracs, alpha=0.5, color="crimson", lw=1)
    if len(clip_fracs) > 20:
        ma_clip = moving_average(clip_fracs, 20)
        ax.plot(range(20, len(clip_fracs) + 1), ma_clip, color="crimson", lw=2,
                label="moving avg")
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Update")
    ax.set_ylabel("Clip Fraction")
    ax.set_title("PPO — Clipping Frequency")
    ax.legend(["clip frac", "moving avg"])
    ax.grid(alpha=0.3)

    # Policy entropy
    ax = axes[2]
    ax.plot(updates, entropies, color="seagreen", lw=1.5, label="entropy")
    ax.set_xlabel("Update")
    ax.set_ylabel("Average Policy Entropy")
    ax.set_title("PPO — Policy Entropy Over Time")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "ppo_results.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    main()
