from itertools import product
from typing import List, Callable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from agents.mb_qvi import MB_QVI
from agents.optimal_oracle import Optimal
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from utils.configuration import load_config_from_args
from utils.utils import plot_occupancies


def main():
    params = load_config_from_args()
    agents = [
        RandomBaseline,
        MB_QVI,
        RF_UCRL,
        BPI_UCRL,
        Optimal,
    ]
    show_occupancies(agents, params)


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
