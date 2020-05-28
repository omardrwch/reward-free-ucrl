"""
Check how many samples are needed before RF-UCRL stops.
"""

from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from agents.base_agent import experiment, BaseAgent
from agents.rf_ucrl import RF_UCRL
from envs.doublechain import DoubleChain, DoubleChainExp


np.random.seed(1253)

# Create parameters
params = {
    "env": DoubleChainExp(5, 0.1),
    "horizon": 10,
    "n_samples_list": np.logspace(2, 6, 30, dtype=np.int32),
    "gamma": 0.99,
    "bonus_scale_factor": 1.0,
    # extra params for RF_UCRL
    "clip": False,
    # n_runs and n_jobs
    "n_runs": 4,
    "n_jobs": 4
}

if __name__=="__main__":
    print("here")

    # Run RF_UCRL
    results = experiment(RF_UCRL, params)
    print(results)

    # Save data
    results.to_csv('rf_sample_complexity.csv')