from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from agents.base_agent import experiment
from agents.mb_qvi import MB_QVI
from agents.optimal_oracle import Optimal
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from envs.chain import Chain
from envs.doublechain import DoubleChain, DoubleChainExp
from envs.gridworld import GridWorld
from utils import plot_error, plot_occupancies

np.random.seed(1253)

# Create parameters
params = {
    # "env": DoubleChainExp(31, 0.1),
    "env": GridWorld(nrows=6, ncols=6),
    "n_samples_list": np.logspace(2, 4, 9, dtype=np.int32),
    "horizon": 16,
    "gamma": 1.0,
    "bonus_scale_factor": 1.0,
    # extra params for RF_UCRL
    "clip": False,
    # n_runs and n_jobs
    "n_runs": 46,
    "n_jobs": 46
}


def estimation_error():
    try:
        data = pd.read_csv('data.csv')
        if data.empty:
            raise FileNotFoundError
    except FileNotFoundError:
        data = pd.DataFrame(columns=['algorithm', 'samples', 'error', 'error-ucb'])

        # Run RandomBaseline
        results = experiment(RandomBaseline, params)
        data = data.append(results, sort=False)

        # Run MB-QVI
        results = experiment(MB_QVI, params)
        data = data.append(results, sort=False)

        # Run RF_UCRL
        results = experiment(RF_UCRL, params)
        data = data.append(results.assign(algorithm="RF-UCRL"), sort=False)

        # Run BPI_UCRL
        results = experiment(BPI_UCRL, params)
        data = data.append(results, sort=False)

        data.to_csv('data.csv')
    plot_error(data)


def show_occupancies(samples=1000):
    agents = [
        RandomBaseline(**params),
        MB_QVI(**params),
        RF_UCRL(**params),
        BPI_UCRL(**params),
        Optimal(**params),
    ]

    def occupancies(agent):
        agent.env.seed(np.random.randint(32768))
        agent.run(samples)
        df = pd.DataFrame({"occupancy": agent.N_sa.sum(axis=1),
                           "state": np.arange(agent.N_sa.shape[0])})
        df["algorithm"] = agent.name
        df["samples"] = samples
        return df

    output = Parallel(n_jobs=params["n_jobs"], verbose=5)(
        delayed(occupancies)(agent) for agent, _ in product(agents, range(params["n_runs"])))
    data = pd.concat(output, ignore_index=True)
    plot_occupancies(data, agents[0].env)


if __name__== "__main__":
    show_occupancies()
    estimation_error()
