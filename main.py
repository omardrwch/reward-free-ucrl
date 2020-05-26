import numpy as np
import matplotlib.pyplot as plt

from agents.base_agent import experiment
from envs.chain import Chain
from agents.mb_qvi import MB_QVI
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from utils import  plot_error, plot_error_upper_bound

np.random.seed(1253)

if __name__=="__main__":
    # Create parameters
    params = {}
    params["env"]            = Chain(10, 0.25)
    params["n_samples_list"] = [100, 250, 500, 1000, 1500, 2000, 5000, 10000, 15000]   # total samples (not per (s,a) ) 
    params["horizon"]        = 10
    params["gamma"]          = 1.0 

    # extra params for RF_UCRL
    params["bonus_scale_factor"] = 1.0
    params["clip"] = True

    # n_runs and n_jobs
    params["n_runs"]         = 20
    params["n_jobs"]         = 4

    # Run RandomBaseline
    errors = experiment(RandomBaseline, params)
    plot_error(params["n_samples_list"], errors, label="RandomBaseline", fignum=1)

    # Run MB-QVI
    errors = experiment(MB_QVI, params)
    plot_error(params["n_samples_list"], errors, label="MB-QVI", fignum=1)

    # Run RF_UCRL with clipping
    params["clip"] = True
    results = experiment(RF_UCRL, params)
    errors, error_ucb = results[..., 0], results[..., 1]
    plot_error(params["n_samples_list"], errors, label="RF-UCRL with clip", fignum=1)
    plot_error_upper_bound(params["n_samples_list"], error_ucb, label="RF-UCRL with clip", fignum=2)

    # Run RF_UCRL without clipping
    params["clip"] = False
    results = experiment(RF_UCRL, params)
    errors, error_ucb = results[..., 0], results[..., 1]
    plot_error(params["n_samples_list"], errors, label="RF-UCRL without clip", fignum=1)
    plot_error_upper_bound(params["n_samples_list"], error_ucb, label="RF-UCRL without clip", fignum=2)
    # Run BPI_UCRL
    errors = experiment(BPI_UCRL, params)
    plot_error(params["n_samples_list"], errors, label="BPI_UCRL", fignum=1)

    plt.show()
