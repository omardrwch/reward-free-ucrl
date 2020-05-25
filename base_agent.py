"""
Baseline for Reward-Free UCRL

Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 

Sample n transitions from each state-action pair to estimate the model \hat{P},
then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function
"""

import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy
from numba import jit


class BaseAgent(object):
    """
        Base agent collecting statistics
    """
    def __init__(self, env, horizon, gamma, **kwargs):
        self.env = deepcopy(env)
        self.env.seed(np.random.randint(32768))      # <--------- important to seed the environment
        self.H = horizon
        self.gamma = gamma
        self.trueR = self.env.mean_R                 # <---------  NOT IN GYM, ATTENTION HERE
        self.trueP = self.env.P                      # <---------  NOT IN GYM, ATTENTION HERE
        self.S = self.env.observation_space.n
        self.A = self.env.action_space.n
        self.P_hat = None
        self.N_sa = None
        self.N_sas = None
        self.trueQ, _ = self.run_value_iteration(self.trueR, self.trueP, self.H, self.gamma)

    def reset(self):
        S, A = self.S, self.A
        self.N_sa = np.zeros((S, A))
        self.N_sas = np.zeros((S, A, S))
        self.P_hat = np.ones((S, A, S)) / S

    def step(self, state, action):
        next_state, _, _, _ = self.env.step(action)
        self.update_model(state, action, next_state)
        return next_state

    def update_model(self, state, action, next_state):
        self.N_sa[state, action] += 1
        self.N_sas[state, action, next_state] += 1
        self.P_hat[state, action, :] = self.N_sas[state, action, :] / self.N_sa[state, action]

    def estimate_value(self):
        """
        :return: Q_hat, V_hat
        """
        return self.run_value_iteration(self.trueR, self.P_hat, self.H, self.gamma)

    def estimation_error(self):
        Q_hat, V_hat = self.estimate_value()
        return np.abs(Q_hat[0] - self.trueQ[0]).max()

    @staticmethod
    @jit(nopython=True)
    def run_value_iteration(R, P, horizon, gamma):
        S, A = R.shape
        V = np.zeros((horizon, S))
        Q = np.zeros((horizon, S, A))
        for hh in range(horizon-1, -1, -1):
            for ss in range(S):
                max_q = -np.inf
                for aa in range(A):
                    q_aa = R[ss, aa]
                    if hh < horizon - 1:
                        q_aa += gamma*P[ss, aa, :].dot(V[hh+1, :])
                    if q_aa > max_q:
                        max_q = q_aa
                    Q[hh, ss, aa] = q_aa
                V[hh, ss] = max_q
        return Q, V

    def run(self, total_samples):
        raise NotImplementedError

    def run_multiple_n(self, n_list):
        return [self.run(n) for n in n_list]


def experiment_worker(agent_class, params):
    agent = agent_class(**params)
    return agent.run_multiple_n(params["n_samples_list"])


def experiment(agent_class, params):
    """
    Run agent in parallel, returns array of dimension (n_runs, len(n_samples_list), *)
    """
    output = Parallel(n_jobs=params["n_jobs"], verbose=5)(
        delayed(experiment_worker)(agent_class, params) for _ in range(params["n_runs"]))
    return np.array(output)
