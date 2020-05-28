r"""
Baseline for Reward-Free UCRL

Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 

Sample n transitions from each state-action pair to estimate the model \hat{P},
then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function
"""
from typing import Tuple, Callable
from joblib import Parallel, delayed
from itertools import product
from copy import deepcopy
from numba import jit
import pandas as pd
import numpy as np

from envs.finitemdp import FiniteMDP


class BaseAgent(object):
    """
        Base agent collecting statistics
    """
    name: str = "Base agent"

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float, **kwargs: dict) -> None:
        self.env = deepcopy(env)
        self.env.seed(np.random.randint(32768))  # <--------- important to seed the environment
        self.H = horizon
        self.gamma = gamma
        self.trueR = self.env.mean_R  # <---------  NOT IN GYM, ATTENTION HERE
        self.trueP = self.env.P  # <---------  NOT IN GYM, ATTENTION HERE
        self.S = self.env.observation_space.n
        self.A = self.env.action_space.n
        self.P_hat = None
        self.N_sa = None
        self.N_sas = None
        self.trueQ, self.trueV = self.run_value_iteration(self.trueR, self.trueP, self.H, self.gamma)

    def reset(self) -> None:
        S, A = self.S, self.A
        self.N_sa = np.zeros((S, A))
        self.N_sas = np.zeros((S, A, S))
        self.P_hat = np.ones((S, A, S)) / S

    def step(self, state: int, action: int) -> int:
        next_state, _, _, _ = self.env.step(action)
        self.update_model(state, action, next_state)
        return next_state

    def update_model(self, state: int, action: int, next_state: int) -> None:
        self.N_sa[state, action] += 1
        self.N_sas[state, action, next_state] += 1
        self.P_hat[state, action, :] = self.N_sas[state, action, :] / self.N_sa[state, action]

    def estimate_value(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Q_hat, V_hat
        """
        return self.run_value_iteration(self.trueR, self.P_hat, self.H, self.gamma)

    def estimation_error(self) -> float:
        initial_state = self.env.reset()
        Q_hat, V_hat = self.estimate_value()
        return np.abs(V_hat[0, initial_state] - self.trueV[0, initial_state])

    @staticmethod
    @jit(nopython=True)
    def run_value_iteration(R: np.ndarray, P: np.ndarray, horizon: int, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        S, A = R.shape
        V = np.zeros((horizon, S))
        Q = np.zeros((horizon, S, A))
        for hh in range(horizon - 1, -1, -1):
            for ss in range(S):
                max_q = -np.inf
                for aa in range(A):
                    q_aa = R[ss, aa]
                    if hh < horizon - 1:
                        q_aa += gamma * P[ss, aa, :].dot(V[hh + 1, :])
                    if q_aa > max_q:
                        max_q = q_aa
                    Q[hh, ss, aa] = q_aa
                V[hh, ss] = max_q
        return Q, V

    def run(self, total_samples: int) -> pd.DataFrame:
        return pd.DataFrame({
            "algorithm": self.name,
            "samples": total_samples,
            "error": self.estimation_error()
        })


def experiment_worker(agent_class: Callable, params: dict, total_samples: int) -> pd.DataFrame:
    agent = agent_class(**params)
    return agent.run(total_samples)


def experiment(agent_class: Callable, params: dict) -> pd.DataFrame:
    """
    Run agent for multiple runs with multiple sample counts, in parallel
    """
    if params["n_jobs"] > 1:
        output = Parallel(n_jobs=params["n_jobs"], verbose=5)(
            delayed(experiment_worker)(agent_class, params, samples)
            for _, samples in product(range(params["n_runs"]), params["n_samples_list"]))
    else:
        output = [experiment_worker(agent_class, params, samples)
                  for _, samples in product(range(params["n_runs"]), params["n_samples_list"])]
    return pd.concat(output, ignore_index=True)
