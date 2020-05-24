"""
Baseline for Reward-Free UCRL

Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 

Sample n transitions from each state-action pair to estimate the model \hat{P},
then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function
"""

import numpy as np
from utils import run_value_iteration
from joblib import Parallel, delayed
from copy import deepcopy


def MB_QVI_experiment(params):
    """
    Run MB_QVI in parallel, returns array of dimension (n_runs, len(n_samples_list)) 
    """
    output = Parallel(n_jobs=params["n_jobs"], verbose=5) \
                     (delayed(MB_QVI_experiment_worker)(params) for ii in range(params["n_runs"]))
    return np.array(output)


def MB_QVI_experiment_worker(params):
    mb_qvi = MB_QVI(params["env"], 
                    params["horizon"], 
                    params["gamma"])
    error_list = mb_qvi.run_multiple_n(params["n_samples_list"])
    return error_list 

class MB_QVI:
    """
    Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 
    
    Sample n transitions from each state-action pair to estimate the model \hat{P},
    then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function

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
        n_samples_per_pair = int(np.ceil(total_samples/(self.S*self.A)))
        # estimate model
        for ss in range(self.S):
            for aa in range(self.A):
                for nn in range(n_samples_per_pair):
                    # sample transition from (ss, aa)
                    self.env.reset(ss)   # put env in state ss     <---------  NOT IN GYM, ATTENTION HERE
                    next_state, _, _, _ = self.env.step(aa)
                    # update counts
                    self.N_sa[ss, aa] += 1
                    self.N_sas[ss, aa, next_state] += 1
                # update P_hat
                self.P_hat[ss, aa, :] = self.N_sas[ss, aa, :] / self.N_sa[ss, aa]
        
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