"""
Reward-Free UCRL (paper)
"""

import numpy as np
from utils import run_value_iteration
from joblib import Parallel, delayed
from copy import deepcopy
from numba import jit


def RF_UCRL_experiment(params):
    """
    Run RF_UCRL in parallel, returns array of dimension (n_runs, len(n_samples_list)) 
    """
    output = Parallel(n_jobs=params["n_jobs"], verbose=5) \
                     (delayed(RF_UCRL_experiment_worker)(params) for ii in range(params["n_runs"]))
    q_errors           = np.array([output[ii][0] for ii in range(len(output))])
    error_upper_bounds = np.array([output[ii][1] for ii in range(len(output))])
    return q_errors, error_upper_bounds



def RF_UCRL_experiment_worker(params):
    rf_ucrl = RF_UCRL(params["env"], 
                      params["horizon"], 
                      params["gamma"], 
                      params["clip"],
                      params["bonus_scale_factor"])
    error_list, error_upper_bound_list = rf_ucrl.run_multiple_n(params["n_samples_list"])
    return error_list, error_upper_bound_list


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
                if (aa == 0 or q_aa > max_q):
                    max_q = q_aa 
                E[hh, ss, aa] = q_aa
            F[hh, ss] = max_q
            if clip:
                F[hh, ss] = min( 2*vmax[hh], F[hh, ss])

class RF_UCRL:
    """   
    :param _env:  environment with discrete state and action spaces
    :param _horizon:
    :param _gamma:
    """
    def __init__(self, _env, _horizon, _gamma, _clip, _bonus_scale_factor):
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

        # other parameters
        self.DELTA = 0.1    # fixed value of delta, for simplicity 
        self.clip  = _clip 
        self.bonus_scale_factor = _bonus_scale_factor

        self.E  = None      # upper bound on error, E(h, s,a) in the paper
        self.F  = None      # F(h, s) = max_a E(h, s, a)

        # compute maximum value function for each step h
        self.vmax = np.zeros(self.H+1)
        for hh in range(self.H-1, -1, -1):
            self.vmax[hh] = 1.0 + self.gamma*self.vmax[hh+1]

    def reset(self):
        S = self.S
        A = self.A
        self.P_hat = np.zeros((S, A, S))
        self.N_sa  = np.zeros((S, A))
        self.N_sas = np.zeros((S, A, S))

        self.E = np.zeros((self.H, S, A))
        self.F = np.zeros((self.H, S))

    def compute_beta(self):
        S, A, H = self.S, self.A, self.H 
        beta    = np.log(2*S*A*H/self.DELTA) + (S-1)*np.log(np.e*( 1+self.N_sa/(S-1)))
        return beta

    def run(self, total_samples):
        self.reset()

        # explore and gather data
        sample_count = 0
        while sample_count < total_samples:
            # run episode
            state = self.env.reset()
            for hh in range(self.H):
                # action = self.env.action_space.sample()
                action = self.E[hh, state,:].argmax()
                next_state, _, _, _ = self.env.step(action)

                # update counts
                self.N_sa[state, action] += 1
                self.N_sas[state, action, next_state] += 1  
                sample_count += 1              

                # update P_hat
                self.P_hat[state, action, :] = self.N_sas[state, action, :] / self.N_sa[state, action]

                # update state
                state = next_state

            # --- finished episode

            # compute bonus
            beta = self.compute_beta()
            n_sa = np.maximum(1, self.N_sa)
            bonus = np.sqrt(beta/n_sa)

            # compute error upper bound
            compute_error_upper_bound(self.E, 
                                      self.F, 
                                      self.P_hat, 
                                      self.H, 
                                      self.gamma, 
                                      bonus, 
                                      self.vmax, 
                                      self.clip, 
                                      self.bonus_scale_factor)

        # run value iteration and compute error
        Q_hat, V_hat = run_value_iteration(self.trueR, self.P_hat, self.H, self.gamma)
        q_error = np.abs(Q_hat[0] - self.trueQ[0]).max()

        initial_state = self.env.reset()
        error_upper_bound = self.F[0, initial_state]  # max_a E[0, initial_state, a]

        return q_error, error_upper_bound, Q_hat, V_hat
    
    def run_multiple_n(self, n_list):
        q_error           = np.zeros(len(n_list))
        error_upper_bound = np.zeros(len(n_list))

        for ii, n in enumerate(n_list):
            error_ii, upper_bound_ii, _, _ = self.run(n)
            q_error[ii] = error_ii
            error_upper_bound[ii] = upper_bound_ii
        return q_error, error_upper_bound
