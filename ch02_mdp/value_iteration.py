"""
ch02 — Dynamic Programming: Value Iteration
Solves GridWorld to optimality.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.env import GridWorld


def value_iteration(env: GridWorld, gamma: float = 0.99, theta: float = 1e-6, max_iter: int = 1000):
    V = np.zeros(env.n_states)
    history = []

    for iteration in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in range(env.n_states):
            row, col = divmod(s, env.width)
            # Terminal states keep zero value
            if (row, col) in [env.goal, env.hole]:
                continue
            action_values = []
            for a in range(env.n_actions):
                q = 0.0
                for prob, s_next, reward, _ in env.transitions(s, a):
                    q += prob * (reward + gamma * V[s_next])
                action_values.append(q)
            V_new[s] = max(action_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        history.append(delta)
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations (delta={delta:.2e})")
            break

    return V, history


def extract_policy(env: GridWorld, V: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        row, col = divmod(s, env.width)
        if (row, col) in [env.goal, env.hole]:
            continue
        action_values = []
        for a in range(env.n_actions):
            q = sum(prob * (reward + gamma * V[s_next])
                    for prob, s_next, reward, _ in env.transitions(s, a))
            action_values.append(q)
        policy[s] = int(np.argmax(action_values))
    return policy


def print_value_table(env: GridWorld, V: np.ndarray):
    print("\n=== Value Function ===")
    for r in range(env.height):
        row_str = []
        for c in range(env.width):
            s = r * env.width + c
            row_str.append(f"{V[s]:+6.3f}")
        print("  ".join(row_str))


def print_policy(env: GridWorld, policy: np.ndarray):
    symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    print("\n=== Optimal Policy ===")
    for r in range(env.height):
        row_str = []
        for c in range(env.width):
            s = r * env.width + c
            if (r, c) == env.goal:
                row_str.append(" G ")
            elif (r, c) == env.hole:
                row_str.append(" X ")
            else:
                row_str.append(f" {symbols[policy[s]]} ")
        print(" | ".join(row_str))


def main():
    env = GridWorld(width=4, height=4)
    gamma = 0.99

    print("=== Value Iteration on 4×4 GridWorld ===")
    print(f"Goal: {env.goal}  Hole: {env.hole}  gamma={gamma}")
    print("\nRunning value iteration …")

    V, history = value_iteration(env, gamma=gamma)
    policy = extract_policy(env, V, gamma=gamma)

    print_value_table(env, V)
    print_policy(env, policy)

    print("\n=== Convergence (first 20 iterations) ===")
    for i, d in enumerate(history[:20]):
        bar = "█" * int(50 * d / (history[0] + 1e-9))
        print(f"  iter {i+1:3d}: delta={d:.6f}  {bar}")

    print("\nFinal env render with policy overlaid:")
    env.reset()
    env.render(policy=policy)


if __name__ == "__main__":
    main()
