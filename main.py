import numpy as np
import matplotlib.pyplot as plt
from envs.chain import Chain
from envs.gridworld import GridWorld
from utils import  plot_error, plot_error_upper_bound
from mb_qvi import MB_QVI_experiment
from random_baseline import RandomBaseline_experiment
from rf_ucrl import RF_UCRL_experiment, RF_UCRL_experiment_worker

np.random.seed(1253)

if __name__=="__main__":
    # Create parameters
    params = {}
    params["env"]            = Chain(10, 0.25)
    params["n_samples_list"] = [100, 500, 1000, 5000, 10000, 15000]   # total samples (not per (s,a) ) 
    params["horizon"]        = 10
    params["gamma"]          = 1.0 

    # extra params for RF_UCRL
    params["bonus_scale_factor"] = 1.0
    params["clip"] = True

    # n_runs and n_jobs
    params["n_runs"]         = 20
    params["n_jobs"]         = 4

    # Run MB_QVI
    error_array = MB_QVI_experiment(params)
    plot_error(params["n_samples_list"], error_array, label="MB_QVI", fignum=1)

    # Run RandomBaseline
    error_array = RandomBaseline_experiment(params)
    plot_error(params["n_samples_list"], error_array, label="RandomBaseline", fignum=1)

    # Run RF_UCRL with clipping
    params["clip"] = True
    q_error_array, error_upper_bound_array = RF_UCRL_experiment(params)
    plot_error(params["n_samples_list"], q_error_array, label="RF-UCRL with clip", fignum=1)
    plot_error_upper_bound(params["n_samples_list"], error_upper_bound_array, label="RF-UCRL with clip", fignum=2)

    # Run RF_UCRL without clipping
    params["clip"] = False
    q_error_array, error_upper_bound_array = RF_UCRL_experiment(params)
    plot_error(params["n_samples_list"], q_error_array, label="RF-UCRL without clip", fignum=1)
    plot_error_upper_bound(params["n_samples_list"], error_upper_bound_array, label="RF-UCRL without clip", fignum=2)


    plt.show()