from itertools import product
import numpy as np
from base_agent import BaseAgent
from utils import kl_upper_bound, max_expectation_under_constraint


class UCRL(BaseAgent):
    """
    UCRL
    """
    DELTA = 0.1

    def __init__(self, env, horizon, gamma, reward_known=False, **kwargs):
        super().__init__(env, horizon, gamma)
        self.reward_known = reward_known
        self.total_reward = None
        self.reward_ucb = None
        self.q_ucb = None
        self.v_ucb = None

    def reset(self):
        super().reset()
        self.total_reward = np.zeros((self.S, self.A))
        self.reward_ucb = np.zeros((self.S, self.A))
        self.q_ucb = np.zeros((self.H, self.S, self.A))
        self.v_ucb = np.zeros((self.H, self.S))

    def step(self, state, action):
        next_state, reward, _, _ = self.env.step(action)
        self.update_model(state, action, next_state)
        self.update_reward(state, action, reward)
        return next_state

    def update_reward(self, state, action, reward):
        self.total_reward[state, action] += reward

    def compute_value_upper_bound(self):
        S, A = self.q_ucb[0, :, :].shape
        threshold = np.log(1/self.DELTA)
        if not self.reward_known:
            for s, a in product(range(S), range(A)):
                self.reward_ucb[s, a] = kl_upper_bound(self.total_reward[s, a], self.N_sa[s, a], threshold=threshold)
        else:
            self.reward_ucb = self.trueR
        for h, s, a in product(range(self.H-1, -1, -1), range(S), range(A)):
            next_v = self.v_ucb[h+1] if h < self.H-1 else np.zeros((S,))
            u_next = self.reward_ucb[s, a] + self.gamma * next_v
            p_plus = max_expectation_under_constraint(u_next, self.P_hat[s, a],
                                                      threshold / np.maximum(self.N_sa[s, a], 1))
            self.q_ucb[h, s, a] = p_plus @ u_next
            self.v_ucb[h, s] = self.q_ucb[h, s].max()

    def run(self, total_samples):
        self.reset()
        # explore and gather data
        sample_count = 0
        print("Running UCRL for ", total_samples)
        while sample_count < total_samples:
            # run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = self.q_ucb[hh, state, :].argmax()
                state = self.step(state, action)

            # Compute upper bound
            self.compute_value_upper_bound()

        error = self.estimation_error()
        return error
