import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True) 
def run_value_iteration(R, P, horizon, gamma):
    S, A = R.shape
    V    = np.zeros((horizon, S))
    Q    = np.zeros((horizon, S, A))
    for hh in range(horizon-1, -1, -1):
        for ss in range(S):
            max_q = 0
            for aa in range(A):
                q_aa = R[ss, aa]
                if hh < horizon - 1:
                    q_aa += gamma*P[ss, aa, :].dot(V[hh+1, :])
                if (aa == 0 or q_aa > max_q):
                    max_q = q_aa 
                Q[hh, ss, aa] = q_aa
            V[hh, ss] = max_q
    return Q, V


def plot_error(xdata, error_array, label, fignum):
    plt.figure(fignum)
    mean_error = error_array.mean(axis=0)
    std_error  = error_array.std(axis=0)
    plt.plot(xdata, mean_error, label=label)
    plt.fill_between(xdata, mean_error-std_error, mean_error+std_error, alpha=0.25)
    plt.legend()
    plt.xlabel("number of samples")
    plt.ylabel("$||\hat{Q}^* - Q^*||_\infty$")
    plt.title("$||\hat{Q}^* - Q^*||_\infty$ versus total number of samples")

def plot_error_upper_bound(xdata, error_array, label, fignum):
    plt.figure(fignum)
    mean_error = error_array.mean(axis=0)
    std_error  = error_array.std(axis=0)
    plt.plot(xdata, mean_error, label=label)
    plt.fill_between(xdata, mean_error-std_error, mean_error+std_error, alpha=0.25)
    plt.legend()
    plt.xlabel("number of samples")
    plt.title("$\max_a E_0(s_1, a)$")
