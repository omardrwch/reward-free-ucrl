from itertools import product
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from envs.finitemdp import FiniteMDP
from utils.utils import kl_upper_bound, max_expectation_under_constraint, random_argmax


class BPI_UCRL(BaseAgent):
    """
    BPI_UCRL
    """
    name: str = "BPI-UCRL"
    DELTA: float = 0.1
    THRESHOLD: float = np.log(1/DELTA)

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float, reward_known: bool = True,
                 transition_support_known: bool = False, **kwargs: dict) -> None:
        super().__init__(env, horizon, gamma)
        self.reward_known = reward_known
        self.transition_support_known = transition_support_known
        self.total_reward = None
        self.reward_ucb, self.reward_lcb = None, None
        self.q_ucb, self.q_lcb = None, None
        self.v_ucb, self.v_lcb = None, None

    def reset(self) -> None:
        super().reset()
        self.P_hat = np.zeros((self.S, self.A, self.S))
        self.total_reward = np.zeros((self.S, self.A))
        self.reward_ucb = np.zeros((self.S, self.A)) if not self.reward_known else self.trueR
        self.reward_lcb = np.zeros((self.S, self.A)) if not self.reward_known else self.trueR
        self.q_ucb = np.zeros((self.H, self.S, self.A))
        self.q_lcb = np.zeros((self.H, self.S, self.A))
        self.v_ucb = np.zeros((self.H, self.S))
        self.v_lcb = np.zeros((self.H, self.S))

    def step(self, state: int, action: int) -> int:
        next_state, reward, _, _ = self.env.step(action)
        self.update_model(state, action, next_state)
        self.update_reward(state, action, reward)
        return next_state

    def update_reward(self, state: int, action: int, reward: int) -> None:
        if not self.reward_known:
            self.total_reward[state, action] += reward
            self.reward_ucb[state, action] = kl_upper_bound(self.total_reward[state, action], self.N_sa[state, action],
                                                            threshold=self.THRESHOLD)
            self.reward_lcb[state, action] = kl_upper_bound(self.total_reward[state, action], self.N_sa[state, action],
                                                            threshold=self.THRESHOLD)

    # @timing
    def compute_value_bounds(self, lower: bool = False) -> None:
        S, A = self.q_ucb[0, :, :].shape
        for h, s, a in product(range(self.H-1, -1, -1), range(S), range(A)):
            # Empirical
            p_next = self.P_hat[s, a]
            # UCB
            next_v = self.v_ucb[h+1] if h < self.H-1 else np.zeros((S,))
            if self.transition_support_known:
                s_next = self.env.get_transition_support(s)
                next_v = next_v[s_next]
                p_next = p_next[s_next]
            p_plus = max_expectation_under_constraint(next_v, p_next,
                                                      self.THRESHOLD / np.maximum(self.N_sa[s, a], 1))
            self.q_ucb[h, s, a] = self.reward_ucb[s, a] + self.gamma * p_plus @ next_v
            self.v_ucb[h, s] = self.q_ucb[h, s].max()
            # LCB
            if lower:
                next_v = self.v_lcb[h+1] if h < self.H-1 else np.zeros((S,))
                if self.transition_support_known:
                    s_next = self.env.get_transition_support(s)
                    next_v = next_v[s_next]
                    p_next = p_next[s_next]
                p_minus = max_expectation_under_constraint(-next_v, p_next,
                                                           self.THRESHOLD / np.maximum(self.N_sa[s, a], 1))
                self.q_lcb[h, s, a] = self.reward_lcb[s, a] + self.gamma * p_minus @ next_v
                self.v_lcb[h, s] = self.q_lcb[h, s].max()

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        np.set_printoptions(precision=2)
        sample_count = 0
        while sample_count < total_samples:
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = random_argmax(self.q_ucb[hh, state, :])
                state = self.step(state, action)
            self.compute_value_bounds()

        self.compute_value_bounds(lower=True)
        initial_state = self.env.reset()
        return pd.DataFrame({
            "algorithm": self.name,
            "samples": total_samples,
            "error": self.estimation_error(),
            "value-ucb": self.v_ucb[0, initial_state],
            "value-lcb": self.v_lcb[0, initial_state],
            "error-ucb": self.v_ucb[0, initial_state] - self.v_lcb[0, initial_state]
        }, index=[0])
