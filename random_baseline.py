"""
Baseline for Reward-Free UCRL

Random exploration policy
"""

import numpy as np
from utils import run_value_iteration
from joblib import Parallel, delayed
from copy import deepcopy

def RandomBaseline_experiment(params):
    """
    Run RandomBaseline in parallel, returns array of dimension (n_runs, len(n_samples_list)) 
    """
    output = Parallel(n_jobs=params["n_jobs"], verbose=5) \
                     (delayed(RandomBaseline_experiment_worker)(params) for ii in range(params["n_runs"]))
    return np.array(output)


def RandomBaseline_experiment_worker(params):
    random_baseline = RandomBaseline(params["env"], 
                                     params["horizon"], 
                                     params["gamma"])
    error_list = random_baseline.run_multiple_n(params["n_samples_list"])
    return error_list 


class RandomBaseline:
    """   
    :param _env:  environment with discrete state and action spaces
    :param _horizon:
    :param _gamma:
    """
    def __init__(self, _env, _horizon, _gamma):
        self.env   = deepcopy(_env)
        self.env.seed(np.random.randint(32768))      # <--------- important to seed the environment
        self.H     = _horizon 
        self.gamma = _gamma
        self.trueR = self.env.mean_R                 # <---------  NOT IN GYM, ATTENTION HERE
        self.trueP = self.env.P                      # <---------  NOT IN GYM, ATTENTION HERE
        self.S     = self.env.observation_space.n 
        self.A     = self.env.action_space.n 
        self.P_hat = None 
        self.N_sa  = None
        self.N_sas = None
        self.trueQ, _ = run_value_iteration(self.trueR, self.trueP, self.H, self.gamma)

    def reset(self):
        S = self.S
        A = self.A
        self.P_hat = np.zeros((S, A, S))
        self.N_sa  = np.zeros((S, A))
        self.N_sas = np.zeros((S, A, S))

    def run(self, total_samples):
        self.reset()

        # explore and gather data
        sample_count = 0
        while sample_count < total_samples:
            state = self.env.reset()
            for hh in range(self.H):
                action = self.env.action_space.sample()
                next_state, _, _, _ = self.env.step(action)
                # update counts
                self.N_sa[state, action] += 1
                self.N_sas[state, action, next_state] += 1  
                sample_count += 1              
                # update state
                state = next_state

        # compute P_hat
        for ss in range(self.S):
            for aa in range(self.A):
                n_sa = max(1, self.N_sa[ss, aa])
                self.P_hat[ss, aa, :] = self.N_sas[ss, aa, :] / n_sa

        # run value iteration and compute error
        Q_hat, V_hat = run_value_iteration(self.trueR, self.P_hat, self.H, self.gamma)
        error = np.abs(Q_hat[0] - self.trueQ[0]).max()

        return error, Q_hat, V_hat
    
    def run_multiple_n(self, n_list):
        error = np.zeros(len(n_list))
        for ii, n in enumerate(n_list):
            error_ii, _, _ = self.run(n)
            error[ii] = error_ii
        return error
