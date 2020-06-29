"""
OptEnt agent

TODO: check clipping
"""
import numpy as np
from numba import jit
from agents.base_agent import BaseAgent
import pandas as pd

from envs.finitemdp import FiniteMDP
from utils.utils import random_argmax


class OptEnt(BaseAgent):
    E: np.ndarray  # upper bound on error, W(h, s,a) in the paper
    F: np.ndarray  # F(h, s) = max_a E(h, s, a)
    name: str = "OptEnt"
    DELTA: float = 0.1    # fixed value of delta, for simplicity

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float, clip: bool, bonus_scale_factor: float,
                 log_all_episodes: bool = False, **kwargs: dict) -> None:
        if gamma != 1.0:
            print("Warning: gamma is NOT used in OptEnt agent and is set to 1.0 by default")
            gamma = 1.0
        
        super().__init__(env, horizon, gamma)

        self.clip = clip
        self.bonus_scale_factor = bonus_scale_factor
        self.log_all_episodes = log_all_episodes

        # compute maximum value function for each step h
        self.v_max = np.zeros(self.H + 1)
        for hh in range(self.H-1, -1, -1):
            self.v_max[hh] = 1 + self.gamma * self.v_max[hh + 1]

    def reset(self) -> None:
        super().reset()
        self.E = np.zeros((self.H, self.S, self.A))
        self.F = np.zeros((self.H, self.S))

    def beta(self) -> float:
        S, A, H = self.S, self.A, self.H 
        return np.log(2*S*A*H/self.DELTA) + (S-1)*np.log(np.e*(1 + self.N_sa/(S-1)))

    @staticmethod
    @jit(nopython=True)
    def compute_error_upper_bound(E: np.ndarray, F: np.ndarray, P_hat: np.ndarray, horizon: int, gamma: float,
                                  bonus: float, vmax: np.ndarray, clip: bool, bonus_scale_factor: float) -> None:
        S, A = E[0, :, :].shape
        for hh in range(horizon-1, -1, -1):
            for ss in range(S):
                max_q = 0
                for aa in range(A):
                    q_aa = bonus_scale_factor*bonus[ss, aa]
                    if hh < horizon - 1:
                        q_aa += (1.0+1.0/horizon)*P_hat[ss, aa, :].dot(F[hh+1, :])
                    if aa == 0 or q_aa > max_q:
                        max_q = q_aa
                    E[hh, ss, aa] = q_aa
                F[hh, ss] = max_q
                if clip:
                    F[hh, ss] = min(horizon, F[hh, ss])

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        sample_count = 0
        samples, errors, ucbs = [], [], []
        while sample_count < total_samples:
            # Run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = random_argmax(self.E[hh, state, :])
                state = self.step(state, action)

            # Compute error upper bound
            # bonus = np.sqrt(self.beta()/np.maximum(1, self.N_sa))
            bonus = 9*(self.H**2.0)*self.beta()/np.maximum(1, self.N_sa)
            self.compute_error_upper_bound(self.E, self.F, self.P_hat, self.H, self.gamma,
                                           bonus, self.v_max, self.clip, self.bonus_scale_factor)

            # Log data
            if self.log_all_episodes or sample_count >= total_samples:
                initial_state = self.env.reset()
                samples.append(sample_count)
                ucbs.append(self.F[0, initial_state])
                errors.append(self.estimation_error())
        return pd.DataFrame({
            "algorithm": [self.name] * len(samples),
            "samples": samples,
            "error": errors,
            "error-ucb": ucbs
        })
