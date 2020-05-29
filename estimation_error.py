from typing import List, Callable

import numpy as np
import pandas as pd

from agents.base_agent import experiment
from agents.mb_qvi import MB_QVI
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from utils.configuration import load_config_from_args
from utils.utils import plot_error


def main():
    params = load_config_from_args()
    if "samples_logspace" in params:
        params["n_samples_list"] = np.logspace(*params["samples_logspace"], dtype=np.int32)
    agents = [
        RandomBaseline,
        MB_QVI,
        RF_UCRL,
        BPI_UCRL,
    ]
    estimation_error(agents, params)


def estimation_error(agents: List[Callable], params: dict) -> None:
    print("--- Estimation error ---")
    try:
        data = pd.read_csv(params["out"] / 'data.csv')
        print("Loaded data from {}".format(params["out"] / 'data.csv'))
        if data.empty:
            raise FileNotFoundError
    except FileNotFoundError:
        data = [experiment(agent, params) for agent in agents]
        data = pd.concat(data, ignore_index=True, sort=False)
        data.to_csv(params["out"] / 'data.csv')
    plot_error(data, out_dir=params["out"])


if __name__ == "__main__":
    main()
