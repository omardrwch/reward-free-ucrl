import importlib
import json
import argparse
from pathlib import Path
from typing import List, Callable

import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed

from agents.base_agent import experiment, BaseAgent
from agents.mb_qvi import MB_QVI
from agents.optimal_oracle import Optimal
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from utils import plot_error, plot_occupancies


def main() -> None:
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", help="Path to the experiment parameters")
    args = parser.parse_args()
    with open(args.parameters) as f:
        params = json.load(f)

    # Seed randomness
    np.random.seed(params["seed"])

    # Make environment
    path = params['env_class'].split("'")[1]
    module_name, class_name = path.rsplit(".", 1)
    env_class = getattr(importlib.import_module(module_name), class_name)
    params["env"] = env_class(**params['env_params'])

    # Make agents
    agents = [
        RandomBaseline,
        MB_QVI,
        RF_UCRL,
        BPI_UCRL,
        Optimal,
    ]

    # Create output directory
    params["out"] = Path(params["out"]) / class_name
    params["out"].mkdir(parents=True, exist_ok=True)

    show_occupancies(agents, params)
    estimation_error(agents, params)


def estimation_error(agents: List[Callable], params: dict) -> None:
    print("--- Estimation error ---")
    if "approximation_samples_logspace" in params:
        params["n_samples_list"] = np.logspace(*params["approximation_samples_logspace"], dtype=np.int32)
    try:
        data = pd.read_csv(params["out"] / 'data.csv')
        print("Loaded data from {}".format(params["out"] / 'data.csv'))
        if data.empty:
            raise FileNotFoundError
    except FileNotFoundError:
        data = [experiment(agent, params) for agent in agents if agent is not Optimal]
        data = pd.concat(data, ignore_index=True, sort=False)
        data.to_csv(params["out"] / 'data.csv')
    plot_error(data, out_dir=params["out"])


def show_occupancies(agents: List[Callable], params: dict) -> None:
    print("--- State occupancies ---")

    def occupancies(agent_class: Callable) -> pd.DataFrame:
        agent = agent_class(**params)
        agent.run(params["occupancies_samples"])
        df = pd.DataFrame({"occupancy": agent.N_sa.sum(axis=1),
                           "state": np.arange(agent.N_sa.shape[0])})
        df["algorithm"] = agent.name
        df["samples"] = params["occupancies_samples"]
        return df
    output = Parallel(n_jobs=params["n_jobs"], verbose=5)(
        delayed(occupancies)(agent) for agent, _ in product(agents, range(params["n_runs"])))
    data = pd.concat(output, ignore_index=True)
    plot_occupancies(data, params["env"], out_dir=params["out"])


if __name__ == "__main__":
    main()
