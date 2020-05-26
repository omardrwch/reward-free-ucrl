"""
Reward-Free UCRL (paper)
"""

import numpy as np
from numba import jit
from agents.base_agent import BaseAgent
import pandas as pd


class RF_UCRL(BaseAgent):
    DELTA = 0.1    # fixed value of delta, for simplicity

    def __init__(self, env, horizon, gamma, clip, bonus_scale_factor, **kwargs):
        super().__init__(env, horizon, gamma)
        self.clip = clip
        self.bonus_scale_factor = bonus_scale_factor
        self.E = None  # upper bound on error, E(h, s,a) in the paper
        self.F = None  # F(h, s) = max_a E(h, s, a)

        # compute maximum value function for each step h
        self.v_max = np.zeros(self.H + 1)
        for hh in range(self.H-1, -1, -1):
            self.v_max[hh] = 1 + self.gamma * self.v_max[hh + 1]

    def reset(self):
        super().reset()
        self.E = np.zeros((self.H, self.S, self.A))
        self.F = np.zeros((self.H, self.S))

    def beta(self):
        S, A, H = self.S, self.A, self.H 
        return np.log(2*S*A*H/self.DELTA) + (S-1)*np.log(np.e*(1 + self.N_sa/(S-1)))

    @staticmethod
    @jit(nopython=True)
    def compute_error_upper_bound(E, F, P_hat, horizon, gamma, bonus, vmax, clip, bonus_scale_factor):
        S, A = E[0, :, :].shape
        for hh in range(horizon-1, -1, -1):
            for ss in range(S):
                max_q = 0
                for aa in range(A):
                    q_aa = bonus_scale_factor*gamma*vmax[hh]*bonus[ss, aa]
                    if hh < horizon - 1:
                        q_aa += gamma*P_hat[ss, aa, :].dot(F[hh+1, :])
                    if aa == 0 or q_aa > max_q:
                        max_q = q_aa
                    E[hh, ss, aa] = q_aa
                F[hh, ss] = max_q
                if clip:
                    F[hh, ss] = min(2*vmax[hh], F[hh, ss])

    def run(self, total_samples):
        self.reset()
        # explore and gather data
        sample_count = 0
        while sample_count < total_samples:
            # run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = self.E[hh, state, :].argmax()
                state = self.step(state, action)

            # Compute error upper bound
            bonus = np.sqrt(self.beta()/np.maximum(1, self.N_sa))
            self.compute_error_upper_bound(self.E, self.F, self.P_hat, self.H, self.gamma,
                                           bonus, self.v_max, self.clip, self.bonus_scale_factor)

        error = self.estimation_error()
        initial_state = self.env.reset()
        error_upper_bound = self.F[0, initial_state]  # max_a E[0, initial_state, a]
        return error, error_upper_bound

    def run_multiple_n(self, n_list):
        errors, error_ucbs = zip(*[self.run(n) for n in n_list])
        return pd.DataFrame({
            "algorithm": [self.__class__.__name__] * len(n_list),
            "samples": n_list,
            "error": errors,
            "error-ucb": error_ucbs
        })
