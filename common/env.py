"""
Common environments for reinforcement learning experiments.
"""
import numpy as np


class GridWorld:
    """
    Simple 4x4 grid world.
    Goal: (3,3), Hole: (1,1)
    Actions: 0=up, 1=down, 2=left, 3=right
    States are linearised: state = row * width + col
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    def __init__(self, width: int = 4, height: int = 4):
        self.width = width
        self.height = height
        self.n_states = width * height
        self.n_actions = 4
        self.goal = (height - 1, width - 1)   # (3, 3)
        self.hole = (1, 1)
        self.start = (0, 0)
        self.state: int = 0

    # ------------------------------------------------------------------ helpers
    def _to_rc(self, state: int):
        return divmod(state, self.width)

    def _to_state(self, row: int, col: int) -> int:
        return row * self.width + col

    # ------------------------------------------------------------------ API
    def reset(self) -> int:
        self.state = self._to_state(*self.start)
        return self.state

    def step(self, action: int):
        row, col = self._to_rc(self.state)
        dr, dc = self.ACTIONS[action]
        new_row = max(0, min(self.height - 1, row + dr))
        new_col = max(0, min(self.width - 1, col + dc))

        next_state = self._to_state(new_row, new_col)
        self.state = next_state

        if (new_row, new_col) == self.goal:
            return next_state, 1.0, True
        if (new_row, new_col) == self.hole:
            return next_state, -1.0, True
        return next_state, -0.01, False

    def render(self, values=None, policy=None):
        """Print the grid.  Optionally overlay value function or policy."""
        lines = []
        for r in range(self.height):
            row_parts = []
            for c in range(self.width):
                s = self._to_state(r, c)
                if (r, c) == self.goal:
                    cell = " G "
                elif (r, c) == self.hole:
                    cell = " X "
                elif s == self.state:
                    cell = " @ "
                elif values is not None:
                    cell = f"{values[s]:+.2f}"
                elif policy is not None:
                    cell = f" {self.ACTION_SYMBOLS[policy[s]]} "
                else:
                    cell = " . "
                row_parts.append(cell)
            lines.append("|".join(row_parts))
        separator = "+".join(["---"] * self.width)
        print(separator)
        print(f"\n{separator}\n".join(lines))
        print(separator)

    # Transition model (used by dynamic programming)
    def transitions(self, state: int, action: int):
        """Returns list of (probability, next_state, reward, done)."""
        row, col = self._to_rc(state)
        if (row, col) in [self.goal, self.hole]:
            return [(1.0, state, 0.0, True)]

        dr, dc = self.ACTIONS[action]
        new_row = max(0, min(self.height - 1, row + dr))
        new_col = max(0, min(self.width - 1, col + dc))
        next_state = self._to_state(new_row, new_col)

        if (new_row, new_col) == self.goal:
            return [(1.0, next_state, 1.0, True)]
        if (new_row, new_col) == self.hole:
            return [(1.0, next_state, -1.0, True)]
        return [(1.0, next_state, -0.01, False)]


class MultiArmedBandit:
    """
    k-armed bandit where each arm has a fixed Bernoulli reward probability.
    """

    def __init__(self, n_arms: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_arms = n_arms
        self.probs = rng.uniform(0.1, 0.9, size=n_arms)
        self.best_arm = int(np.argmax(self.probs))
        self.best_prob = float(self.probs[self.best_arm])

    def pull(self, arm: int) -> float:
        """Return 0 or 1 sampled from the arm's Bernoulli distribution."""
        return float(np.random.random() < self.probs[arm])

    def expected_reward(self, arm: int) -> float:
        return float(self.probs[arm])
