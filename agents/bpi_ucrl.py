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
    BETA: float = np.log(1/DELTA)

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float, reward_known: bool = True,
                 transition_support_known: bool = False, log_all_episodes: bool = False, **kwargs: dict) -> None:
        super().__init__(env, horizon, gamma)
        self.reward_known = reward_known
        self.transition_support_known = transition_support_known
        self.log_all_episodes = log_all_episodes
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
                                                            threshold=self.beta())
            self.reward_lcb[state, action] = kl_upper_bound(self.total_reward[state, action], self.N_sa[state, action],
                                                            threshold=self.beta())

    def beta(self):
        S, A, H = self.S, self.A, self.H
        return np.log(2*S*A*H/self.DELTA) + (S-1)*np.log(np.e*(1 + self.N_sa/(S-1)))

    # @timing
    def compute_value_bounds(self, lower: bool = False) -> None:
        S, A = self.q_ucb[0, :, :].shape
        bonus = self.beta()/np.maximum(1, self.N_sa)
        for h, s, a in product(range(self.H-1, -1, -1), range(S), range(A)):
            # Empirical
            p_next = self.P_hat[s, a]
            self.q_ucb[h, s, a] = self.reward_ucb[s, a]
            if h < self.H-1:
                # UCB
                next_v = self.v_ucb[h+1]
                if self.transition_support_known:
                    s_next = self.env.get_transition_support(s)
                    next_v = next_v[s_next]
                    p_next = p_next[s_next]
                p_plus = max_expectation_under_constraint(next_v, p_next, bonus[s, a])
                self.q_ucb[h, s, a] += self.gamma * p_plus @ next_v
            self.v_ucb[h, s] = self.q_ucb[h, s].max()
            # LCB
            if lower:
                self.q_lcb[h, s, a] = self.reward_lcb[s, a]
                if h < self.H-1:
                    next_v = self.v_lcb[h+1]
                    if self.transition_support_known:
                        next_v = next_v[s_next]
                    p_minus = max_expectation_under_constraint(-next_v, p_next, bonus[s, a])
                    self.q_lcb[h, s, a] += self.gamma * p_minus @ next_v
                self.v_lcb[h, s] = self.q_lcb[h, s].max()

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        samples, errors, ucbs = [], [], []
        sample_count = 0
        while sample_count < total_samples:
            # Run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = random_argmax(self.q_ucb[hh, state, :])
                state = self.step(state, action)

            # Compute upper (and lower?) bound
            log_this_episode = self.log_all_episodes or sample_count >= total_samples
            self.compute_value_bounds(lower=log_this_episode)

            # Log results
            if log_this_episode:
                initial_state = self.env.reset()
                samples.append(sample_count)
                ucbs.append(self.v_ucb[0, initial_state] - self.v_lcb[0, initial_state])
                errors.append(self.estimation_error())
        return pd.DataFrame({
            "algorithm": [self.name] * len(samples),
            "samples": samples,
            "error": errors,
            "error-ucb": ucbs
        })
