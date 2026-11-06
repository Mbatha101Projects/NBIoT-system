

import numpy as np
import random
from dataclasses import dataclass


@dataclass
class QLearningConfig:
    """All tunable hyperparameters in one place, so hyperparameter_search.py
    can sweep over them without touching the agent's logic."""
    alpha: float = 0.1          # learning rate
    gamma: float = 0.9          # discount factor
    epsilon_start: float = 1.0  # initial exploration rate (100% random at first)
    epsilon_min: float = 0.05   # exploration never fully stops — keeps
                                  # adapting if conditions later look new
    epsilon_decay: float = 0.995  # multiplied onto epsilon after each episode


class QLearningAgent:
    """
    A tabular Double Q-learning agent (see module docstring for why
    Double, not plain, Q-learning is used).


    """

    def __init__(self, n_states: int, n_actions: int,
                 config: QLearningConfig = None, seed: int = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config or QLearningConfig()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # TWO independent Q-tables — the core of Double Q-learning.
        # Both initialized to zero, exactly as the single-table version was.
        self.q_table_a = np.full((n_states, n_actions), 2.0, dtype=np.float64)
        self.q_table_b = np.full((n_states, n_actions), 2.0, dtype=np.float64)

        self.epsilon = self.config.epsilon_start

        # Tracking for diagnostics — how many times each (state,action)
        # pair has been visited (across BOTH tables combined), useful for
        # confirming the agent actually explored the full space.
        self.visit_counts = np.zeros((n_states, n_actions), dtype=np.int64)

    @property
    def q_table(self) -> np.ndarray:
        """
        Read-only combined estimate: the average of the two tables.
        This is what choose_action() and policy_summary() use, and what
        external code (train.py, evaluate.py, or ad-hoc diagnostics)
        should read if it wants "the agent's current best estimate" —
        exactly the same role self.q_table played in the single-table
        version, so no calling code needed to change.
        """
        return (self.q_table_a + self.q_table_b) / 2.0

    def choose_action(self, state: int, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection, using the COMBINED (averaged)
        estimate from both tables — standard practice for the behaviour
        policy in Double Q-learning.

        If greedy=True, ALWAYS pick the best-known action (used during
        evaluation, after training is complete — no more exploring).
        If greedy=False, pick randomly with probability epsilon,
        otherwise pick the best-known action (used during training).
        """
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        # Exploit: pick the action with the highest COMBINED value for
        # this state. Ties are broken randomly (np.argmax always picks
        # the FIRST max, which would bias the agent toward low action
        # indices early in training when many entries are still tied
        # at zero — random tie-breaking avoids that bias).
        row = self.q_table[state]
        max_value = row.max()
        best_actions = np.flatnonzero(row == max_value)
        return int(self.np_rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float,
              next_state: int, done: bool) -> None:
        """
        The Double Q-learning update — called once per environment step.

        With 50% probability each, EITHER Q_A is updated (selecting the
        best next action FROM Q_A, but evaluating its value using Q_B),
        OR the reverse. This decoupling of "which action looks best" from
        "how good is that action" is what removes the maximization bias
        that plain single-table Q-learning suffers from (see module
        docstring for the full explanation and the evidence that led to
        this change).
        """
        self.visit_counts[state, action] += 1

        update_a = self.np_rng.random() < 0.5

        if update_a:
            current_q = self.q_table_a[state, action]
            if done:
                target = reward
            else:
                # Action selection from Q_A, value lookup from Q_B —
                # the core decoupling step.
                best_next_action = int(np.argmax(self.q_table_a[next_state]))
                target = reward + self.config.gamma * self.q_table_b[next_state, best_next_action]
            self.q_table_a[state, action] = current_q + self.config.alpha * (target - current_q)
        else:
            current_q = self.q_table_b[state, action]
            if done:
                target = reward
            else:
                best_next_action = int(np.argmax(self.q_table_b[next_state]))
                target = reward + self.config.gamma * self.q_table_a[next_state, best_next_action]
            self.q_table_b[state, action] = current_q + self.config.alpha * (target - current_q)

    def end_episode(self) -> None:
        """Call once per completed episode — decays the exploration rate
        so the agent gradually shifts from exploring to exploiting as it
        learns more, but never reaches exactly zero (epsilon_min floor),
        so it retains some ability to notice if conditions it hasn't seen
        before turn out to behave differently than expected."""
        self.epsilon = max(self.config.epsilon_min,
                          self.epsilon * self.config.epsilon_decay)

    def save(self, path: str) -> None:
        """
        Save BOTH tables to a single file, so training doesn't need to
        be re-run every time the agent is used for evaluation. Stored as
        a stacked array of shape (2, n_states, n_actions) so the file
        format/path stays a single .npy file — no change needed to any
        code that calls agent.save(path) with a plain .npy path.
        """
        stacked = np.stack([self.q_table_a, self.q_table_b])
        np.save(path, stacked)

    def load(self, path: str) -> None:
        """Load a previously-trained pair of Q-tables."""
        stacked = np.load(path)
        assert stacked.shape == (2, self.n_states, self.n_actions), \
            f"Loaded table shape {stacked.shape} does not match " \
            f"expected (2, {self.n_states}, {self.n_actions}) — this file " \
            f"may have been saved by the older single-table agent version; " \
            f"re-run train.py to regenerate it with Double Q-learning."
        self.q_table_a = stacked[0]
        self.q_table_b = stacked[1]

    def policy_summary(self, state_names: list = None,
                       action_names: list = None) -> str:
        """
        Print the agent's learned policy as a readable table: for each
        state, which action does it consider best (using the COMBINED
        estimate)? This is the interpretability payoff of tabular
        Q-learning mentioned in the project's design justification — the
        entire learned strategy can be read directly, not just queried.
        """
        combined = self.q_table
        lines = []
        for s in range(self.n_states):
            best_a = int(np.argmax(combined[s]))
            s_label = state_names[s] if state_names else f"state_{s}"
            a_label = action_names[best_a] if action_names else f"action_{best_a}"
            confidence = combined[s, best_a]
            lines.append(f"  {s_label:30s} -> {a_label:20s}  (Q={confidence:+.4f})")
        return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test: confirm the agent can be created, choose actions, and
    # update BOTH tables without error, using dummy state/action sizes
    # (full integration with environment.py happens in train.py).
    agent = QLearningAgent(n_states=12, n_actions=42, seed=42)

    print(f"Q-table A shape: {agent.q_table_a.shape}")
    print(f"Q-table B shape: {agent.q_table_b.shape}")
    print(f"Combined q_table property shape: {agent.q_table.shape}")
    print(f"Initial epsilon: {agent.epsilon}")
    print()

    before_a = agent.q_table_a.copy()
    before_b = agent.q_table_b.copy()
    for _ in range(200):
        s = agent.rng.randrange(12)
        a = agent.choose_action(s)
        r = agent.rng.uniform(-0.5, 0.5)
        s_next = agent.rng.randrange(12)
        agent.update(s, a, r, s_next, done=False)
    agent.end_episode()

    a_changed = not np.array_equal(before_a, agent.q_table_a)
    b_changed = not np.array_equal(before_b, agent.q_table_b)
    print(f"Q-table A changed after 200 updates: {a_changed}")
    print(f"Q-table B changed after 200 updates: {b_changed}")
    print(f"(Both should be True — confirms updates are genuinely being "
          f"split roughly 50/50 between the two tables, not always hitting one)")
    print(f"Epsilon after 1 episode: {agent.epsilon:.4f} "
          f"(should be slightly below {agent.config.epsilon_start})")
    print(f"Total visits recorded: {agent.visit_counts.sum()} (should be 200)")
    print()

    # Round-trip save/load test
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_qtable.npy")
        agent.save(path)
        agent2 = QLearningAgent(n_states=12, n_actions=42, seed=1)
        agent2.load(path)
        match_a = np.array_equal(agent.q_table_a, agent2.q_table_a)
        match_b = np.array_equal(agent.q_table_b, agent2.q_table_b)
        print(f"Save/load round-trip — table A matches: {match_a}")
        print(f"Save/load round-trip — table B matches: {match_b}")