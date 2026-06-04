"""
ch11 — Function Approximation for Reinforcement Learning

Methods
-------
1. Tabular          — baseline (exact value table)
2. Linear + Poly    — quadratic polynomial features
3. Linear + RBF     — Radial Basis Function features
4. GTD2             — Gradient TD (off-policy stable)

Environment: 10×10 GridWorld.
Plots: convergence curves + learned value-function heatmap.
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


# ═══════════════════════════════════════ Feature extractors ═════════════════════

class TabularFeatures:
    """One-hot encoding — recovers exact tabular representation."""
    def __init__(self, n_states: int):
        self.n_features = n_states

    def __call__(self, state: int) -> np.ndarray:
        phi = np.zeros(self.n_features)
        phi[state] = 1.0
        return phi


class PolynomialFeatures:
    """
    Quadratic polynomial features on (row/H, col/W) coordinates.
    Features: [1, r, c, r², rc, c²]
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.n_features = 6

    def __call__(self, state: int) -> np.ndarray:
        r = (state // self.width) / max(self.height - 1, 1)
        c = (state % self.width)  / max(self.width - 1, 1)
        return np.array([1.0, r, c, r * r, r * c, c * c])


class RBFFeatures:
    """
    Radial Basis Function features.
    Centers placed on a regular grid; γ controls width.
    """
    def __init__(self, width: int, height: int,
                 n_centers: int = 25, gamma: float = 2.0):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.n_features = n_centers

        # Place centers on a regular grid in normalized [0,1]² space
        side = int(np.ceil(np.sqrt(n_centers)))
        cx = np.linspace(0, 1, side)
        cy = np.linspace(0, 1, side)
        grid = np.array([[x, y] for x in cx for y in cy])
        self.centers = grid[:n_centers]   # (n_centers, 2)

    def __call__(self, state: int) -> np.ndarray:
        r = (state // self.width) / max(self.height - 1, 1)
        c = (state % self.width)  / max(self.width - 1, 1)
        x = np.array([r, c])
        diffs = self.centers - x           # (n_centers, 2)
        dist2 = (diffs ** 2).sum(axis=1)   # (n_centers,)
        return np.exp(-self.gamma * dist2)


# ═══════════════════════════════════════ Value estimators ═══════════════════════

class LinearValueFunction:
    """
    V(s) = w^T φ(s)
    Updated via semi-gradient TD(0).
    """
    def __init__(self, feature_fn, alpha: float = 0.01):
        self.phi = feature_fn
        self.w = np.zeros(feature_fn.n_features)
        self.alpha = alpha

    def value(self, state: int) -> float:
        return float(self.w @ self.phi(state))

    def update(self, state: int, target: float):
        phi_s = self.phi(state)
        error = target - self.w @ phi_s
        self.w += self.alpha * error * phi_s

    def values_all(self, n_states: int) -> np.ndarray:
        return np.array([self.value(s) for s in range(n_states)])


class GTD2ValueFunction:
    """
    Gradient TD2 (Sutton et al., 2009) — provably convergent off-policy TD.
    Two weight vectors: w (value weights) and v (auxiliary weights).
    Update:
        δ = r + γ φ(s')^T w − φ(s)^T w
        e = φ(s)
        v ← v + β (δ − e^T v) e          # auxiliary
        w ← w + α (δ e − γ φ(s') e^T v)  # main
    """
    def __init__(self, feature_fn, alpha: float = 0.01, beta: float = 0.001,
                 gamma: float = 0.99):
        self.phi = feature_fn
        self.n = feature_fn.n_features
        self.w = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def value(self, state: int) -> float:
        return float(self.w @ self.phi(state))

    def update(self, state: int, reward: float, next_state: int, done: bool):
        phi_s  = self.phi(state)
        phi_s2 = self.phi(next_state) if not done else np.zeros(self.n)
        td_error = reward + self.gamma * (self.w @ phi_s2) - (self.w @ phi_s)
        # Auxiliary update
        self.v += self.beta * (td_error - float(phi_s @ self.v)) * phi_s
        # Main update
        self.w += self.alpha * (td_error * phi_s - self.gamma * phi_s2 * float(phi_s @ self.v))

    def values_all(self, n_states: int) -> np.ndarray:
        return np.array([self.value(s) for s in range(n_states)])


# ═══════════════════════════════════════ Policy evaluation ══════════════════════

def random_policy(env: GridWorld, state: int) -> int:
    return int(np.random.randint(env.n_actions))


def run_policy_eval(vf, env: GridWorld, n_episodes: int = 2000,
                    gamma: float = 0.99, is_gtd2: bool = False) -> list[float]:
    """
    Evaluate a random policy; return per-episode MSE against Monte-Carlo returns.
    """
    mse_curve: list[float] = []

    for _ in range(n_episodes):
        s = env.reset()
        trajectory: list[tuple[int, float]] = []
        for _ in range(200):
            a = random_policy(env, s)
            s2, r, done = env.step(a)
            trajectory.append((s, r))
            if is_gtd2:
                vf.update(s, r, s2, done)
            else:
                # Semi-gradient TD(0)
                target = r + gamma * vf.value(s2) * (1 - int(done))
                vf.update(s, target)
            s = s2
            if done:
                break

        # Compute MSE vs Monte-Carlo return for visited states
        G = 0.0
        mc_returns: list[tuple[int, float]] = []
        for st, rw in reversed(trajectory):
            G = rw + gamma * G
            mc_returns.append((st, G))
        errors = [(vf.value(st) - G_st) ** 2 for st, G_st in mc_returns]
        mse_curve.append(float(np.mean(errors)) if errors else 0.0)

    return mse_curve


# ═══════════════════════════════════════ Visualization ══════════════════════════

def _smooth(x: list[float], k: int = 50) -> np.ndarray:
    arr = np.array(x)
    kernel = np.ones(k) / k
    if len(arr) < k:
        return arr
    return np.convolve(arr, kernel, mode="valid")


def plot_heatmaps(vfs: dict[str, object], env: GridWorld, out_dir: str) -> None:
    n = len(vfs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, vf) in zip(axes, vfs.items()):
        vals = vf.values_all(env.n_states).reshape(env.height, env.width)
        im = ax.imshow(vals, cmap="RdYlGn", aspect="auto")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Col")
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, fraction=0.046)
        # Mark goal and hole
        goal_r, goal_c = env.goal
        hole_r, hole_c = env.hole
        ax.text(goal_c, goal_r, "G", ha="center", va="center",
                fontsize=12, fontweight="bold", color="black")
        ax.text(hole_c, hole_r, "X", ha="center", va="center",
                fontsize=12, fontweight="bold", color="black")

    plt.suptitle("Learned Value Function Heatmaps (10×10 GridWorld)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "linear_fa_heatmaps.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmaps saved → {path}")


# ═══════════════════════════════════════ Main ═══════════════════════════════════

def main():
    np.random.seed(42)
    env = GridWorld(width=10, height=10)
    n_episodes = 2000
    gamma = 0.99

    print("=" * 60)
    print("Function Approximation on 10×10 GridWorld")
    print("=" * 60)

    w, h = env.width, env.height
    feature_fns = {
        "Tabular":     TabularFeatures(env.n_states),
        "Poly (deg2)": PolynomialFeatures(w, h),
        "RBF (n=25)":  RBFFeatures(w, h, n_centers=25, gamma=3.0),
    }

    mse_curves: dict[str, list[float]] = {}
    trained_vfs: dict[str, object] = {}
    results: dict[str, dict] = {}

    for name, feat_fn in feature_fns.items():
        print(f"  Training {name} …", end=" ", flush=True)
        vf = LinearValueFunction(feat_fn, alpha=0.05)
        mses = run_policy_eval(vf, env, n_episodes=n_episodes, gamma=gamma)
        mse_curves[name] = mses
        trained_vfs[name] = vf
        avg_mse = float(np.mean(mses[-200:]))
        results[name] = {"avg_last200_mse": round(avg_mse, 6)}
        print(f"avg MSE (last 200 ep) = {avg_mse:.6f}")

    # GTD2
    print("  Training GTD2 …", end=" ", flush=True)
    gtd2_feat = RBFFeatures(w, h, n_centers=25, gamma=3.0)
    vf_gtd2 = GTD2ValueFunction(gtd2_feat, alpha=0.01, beta=0.001, gamma=gamma)
    mses_gtd2 = run_policy_eval(vf_gtd2, env, n_episodes=n_episodes,
                                gamma=gamma, is_gtd2=True)
    mse_curves["GTD2 (RBF)"] = mses_gtd2
    trained_vfs["GTD2 (RBF)"] = vf_gtd2
    avg_mse = float(np.mean(mses_gtd2[-200:]))
    results["GTD2"] = {"avg_last200_mse": round(avg_mse, 6)}
    print(f"avg MSE (last 200 ep) = {avg_mse:.6f}")

    # ── Convergence plot ─────────────────────────────────────────────────
    smooth_k = 50
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (name, mses) in enumerate(mse_curves.items()):
        smoothed = _smooth(mses, smooth_k)
        ax.plot(smoothed, label=name, color=colors[j % len(colors)])
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"MSE vs MC returns (smoothed k={smooth_k})")
    ax.set_title("Function Approximation Convergence — 10×10 GridWorld")
    ax.legend()
    ax.grid(alpha=0.25)
    png = os.path.join(OUT_DIR, "linear_fa_convergence.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Convergence plot saved → {png}")

    # ── Heatmaps ─────────────────────────────────────────────────────────
    plot_heatmaps(trained_vfs, env, OUT_DIR)

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "linear_fa_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {json_path}")


if __name__ == "__main__":
    main()
