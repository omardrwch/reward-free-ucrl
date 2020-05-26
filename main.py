import numpy as np
import pandas as pd

from agents.base_agent import experiment
from agents.mb_qvi import MB_QVI
from agents.optimal_oracle import Optimal
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from envs.chain import Chain
from envs.doublechain import DoubleChain, DoubleChainExp
from utils import plot_error, plot_error_upper_bound

np.random.seed(1253)

# Create parameters
params = {}
params["env"]            = DoubleChainExp(27, 0.25)
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

    # # Run RF_UCRL with clipping
    # params["clip"] = True
    # results = experiment(RF_UCRL, params)
    # data = data.append(results.assign(algorithm="RF-UCRL with clip"), sort=False)

    # Run RF_UCRL without clipping
    params["clip"] = False
    results = experiment(RF_UCRL, params)
    data = data.append(results.assign(algorithm="RF-UCRL"), sort=False)

    # Run BPI_UCRL
    results = experiment(BPI_UCRL, params)
    data = data.append(results, sort=False)

    data.to_csv('data.csv')
    plot_error(data)


def show_occupations(samples=1000, runs=20):
    from matplotlib import pyplot as plt
    import seaborn as sns
    del params["clip"]
    agents = {
        "Uniform": RandomBaseline(**params),
        "Optimal policy": Optimal(**params),
        "MB-QVI": MB_QVI(**params),
        # "RF_UCRL with clip": RF_UCRL(**params, clip=True),
        "RF-UCRL": RF_UCRL(**params, clip=False),
        "BPI-UCRL": BPI_UCRL(**params),
    }

    data = pd.DataFrame(columns=["algorithm", "samples", "states", "occupations"])
    for _ in range(runs):
        for name, agent in agents.items():
            agent.run(samples)
            df = pd.DataFrame({"occupations": agent.N_sa.sum(axis=1),
                               "states": np.arange(agent.N_sa.shape[0])})
            df["algorithm"] = name
            df["samples"] = samples
            data = pd.concat([df, data], sort=False)
    plt.figure("occupations")
    sns.lineplot(x="states", y="occupations", hue="algorithm", data=data)
    plt.show()


if __name__== "__main__":
    show_occupations()
    # estimation_error()
