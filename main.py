import numpy as np
import pandas as pd

from agents.base_agent import experiment
from agents.mb_qvi import MB_QVI
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from envs.chain import Chain
from envs.doublechain import DoubleChain
from utils import plot_error, plot_error_upper_bound

np.random.seed(1253)

# Create parameters
params = {}
params["env"]            = DoubleChain(31, 0.25)
params["n_samples_list"] = [100, 250, 500, 1000, 1500, 2000, 5000, 10000, 15000]   # total samples (not per (s,a) )
params["horizon"]        = 15
params["gamma"]          = 1.0

# extra params for RF_UCRL
params["bonus_scale_factor"] = 1.0
params["clip"] = True

# n_runs and n_jobs
params["n_runs"]         = 46
params["n_jobs"]         = 46

def estimation_error():
    data = pd.DataFrame(columns=['algorithm', 'samples', 'error', 'error-ucb'])

    # Run RandomBaseline
    results = experiment(RandomBaseline, params)
    data = data.append(results, sort=False)

    # Run MB-QVI
    results = experiment(MB_QVI, params)
    data = data.append(results, sort=False)

    # Run RF_UCRL with clipping
    params["clip"] = True
    results = experiment(RF_UCRL, params)
    data = data.append(results.assign(algorithm="RF-UCRL with clip"), sort=False)

    # Run RF_UCRL without clipping
    params["clip"] = False
    results = experiment(RF_UCRL, params)
    data = data.append(results.assign(algorithm="RF-UCRL without clip"), sort=False)

    # Run BPI_UCRL
    results = experiment(BPI_UCRL, params)
    data = data.append(results, sort=False)

    data.to_csv('data.csv')
    plot_error(data)


def exploration(samples=1000):
    agent = RandomBaseline(**params)
    agent.run(samples)
    from matplotlib import pyplot as plt
    plt.imshow(agent.N_sa.sum(axis=1, keepdims=True).T)
    plt.show()


if __name__=="__main__":
    estimation_error()
    exploration()