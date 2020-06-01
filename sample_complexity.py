"""
Check how many samples are needed before RF-UCRL stops.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator

from agents.bpi_ucrl import BPI_UCRL
from utils.configuration import load_config_from_args
from agents.base_agent import experiment
from agents.rf_ucrl import RF_UCRL


def main():
    params = load_config_from_args()
    if "complexity_samples_logspace" in params:
        params["n_samples_list"] = np.logspace(*params["complexity_samples_logspace"], dtype=np.int32)
    if "complexity_n_runs" in params:
        params["n_runs"] = params["complexity_n_runs"]
    compute(params)
    plot(params)


def compute(params: dict) -> None:
    path = params["out"] / 'rf_sample_complexity.csv'
    if not path.exists():
        data = experiment(RF_UCRL, params)
        data.to_csv(path)
    bpi_path = params["out"] / 'bpi_sample_complexity.csv'
    if not bpi_path.exists():
        data = experiment(BPI_UCRL, params)
        data.to_csv(bpi_path)


def plot(params: dict) -> None:
    # extract data
    data_rf = pd.read_csv(params["out"] / 'rf_sample_complexity.csv')
    data_bpi = pd.read_csv(params["out"] / 'bpi_sample_complexity.csv')
    data = pd.concat([data_rf, data_bpi], sort=False, ignore_index=True)
    data["error"] /= params["horizon"]
    data["error-ucb"] /= params["horizon"]

    plot_bins(data, out_dir=params["out"])
    plot_error(data, out_dir=params["out"])
    plt.show()


def plot_bins(data, out_dir):
    # plot with number of samples required for each epsilon
    f, ax = plt.subplots()
    data_bins = data.copy()
    data_bins = data_bins[data_bins["error-ucb"].between(0.0, 1.0)]
    error_bins = np.linspace(0, 1.0, 10)
    for ii, ee in enumerate(error_bins[:-1]):
        ee_min = ee
        ee_max = error_bins[ii+1]-1e-8
        data_bins.loc[data_bins["error-ucb"].between(ee, ee_max), "error-ucb"] = \
            int(100*(ee_min+ee_max)/2.0)/100
    ax.set(xscale="linear", yscale="log")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="all"))
    ax.yaxis.grid(True, which='minor', linestyle='--')
    sns.barplot(x="error-ucb", y="samples", data=data_bins,
                errcolor="red", hue="algorithm", ax=ax)
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("Number of samples")
    plt.savefig(out_dir / "error-bins.png")
    plt.savefig(out_dir / "error-bins.pdf")


def plot_error(data, out_dir):
    """ plot error and error UCB"""
    plt.figure()
    f, ax = plt.subplots()
    ax.set(xscale="log", yscale="log")
    for name, df in data.groupby('algorithm'):
        sns.lineplot(x="samples", y="error", data=df, ax=ax, label="Error (empirical) of " + name)
        sns.lineplot(x="samples", y="error-ucb", data=df, ax=ax, label="Error (UCB) of " + name)
    plt.xlabel("Number of samples")
    plt.ylabel("Error (normalized by $H$)")
    # plt.title(r"$\max_a E_0(s_1, a) / H$")
    plt.savefig(out_dir / "error-samples.png")
    plt.savefig(out_dir / "error-samples.pdf")


if __name__ == "__main__":
    main()
