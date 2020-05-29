"""
Check how many samples are needed before RF-UCRL stops.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator

from utils.configuration import load_config_from_args
from agents.base_agent import experiment
from agents.rf_ucrl import RF_UCRL


def main():
    params = load_config_from_args()
    if "complexity_samples_logspace" in params:
        params["n_samples_list"] = np.logspace(*params["complexity_samples_logspace"], dtype=np.int32)
    if "complexity_n_runs" in params:
        params["n_runs"] = params["complexity_samples_logspace"]
    compute(params)
    plot(params)


def compute(params: dict) -> None:
    path = params["out"] / 'rf_sample_complexity.csv'
    if not path.exists():
        results = experiment(RF_UCRL, params)
        results.to_csv(path)


def plot(params: dict) -> None:
    # extract data
    data = pd.read_csv(params["out"] / 'rf_sample_complexity.csv')
    data["error"] /= params["horizon"]
    data["error-ucb"] /= params["horizon"]

    plot_bins(data, out_dir=params["out"])
    plot_error(data, out_dir=params["out"])
    plt.show()


def plot_bins(data, out_dir):
    # plot with number of samples required for each epsilon
    data_bins = data.copy()
    data_bins = data_bins[data_bins["error-ucb"].between(0.0, 1.0)]
    error_bins = np.linspace(0.1, 1.0, 10)
    for ii, ee in enumerate(error_bins[:-1]):
        ee_min = ee
        ee_max = error_bins[ii+1]-1e-8
        data_bins.loc[data_bins["error-ucb"].between(ee, ee_max), "error-ucb"] = \
            int(100*(ee_min+ee_max)/2.0)/100

    # plt.figure("stopping times")
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    f, ax = plt.subplots()
    ax.set(xscale="linear", yscale="log")
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="all"))
    ax.yaxis.grid(True, which='minor', linestyle='--')
    sns.barplot(x="error-ucb", y="samples", data=data_bins, palette="Blues_d", errcolor="red")
    plt.xlabel("$\epsilon$")
    plt.savefig(out_dir / "error-bins.png")
    plt.savefig(out_dir / "error-bins.pdf")


def plot_error(data, out_dir):
    """ plot error and error UCB"""
    plt.figure()
    f, ax = plt.subplots()
    ax.set(xscale="log", yscale="log")
    sns.lineplot(x="samples", y="error", data=data, ax=ax, label="Error (empirical)")
    sns.lineplot(x="samples", y="error-ucb", data=data, ax=ax, label="Error (UCB)")
    plt.xlabel("Number of samples")
    plt.ylabel("Error (normalized by $H$)")
    plt.title(r"$\max_a E_0(s_1, a) / H$")
    plt.savefig(out_dir / "error-samples.png")
    plt.savefig(out_dir / "error-samples.pdf")


if __name__ == "__main__":
    main()
