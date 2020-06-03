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
    path = params["out"] / 'rf_sample_complexity_{}.csv'.format(params["complexity_samples_logspace"][1])
    if not path.exists():
        data = experiment(RF_UCRL, params)
        data.to_csv(path)
    bpi_path = params["out"] / 'bpi_sample_complexity_{}.csv'.format(params["complexity_samples_logspace"][1])
    if not bpi_path.exists():
        data = experiment(BPI_UCRL, params)
        data.to_csv(bpi_path)


def plot(params: dict) -> None:
    # extract data
    data_bpi = pd.read_csv(params["out"] / 'bpi_sample_complexity_6_2.csv'.format(params["complexity_samples_logspace"][1]))
    data_bpi["epsilon"] = data_bpi["error-ucb"]
    print("Loaded BPI data")
    data_rf = pd.read_csv(params["out"] / 'rf_sample_complexity_8.csv'.format(params["complexity_samples_logspace"][1]))
    data_rf["epsilon"] = data_rf["error-ucb"] * 2
    # data_rf = data_rf.iloc[::20, :]
    print("Loaded RF data")
    data = pd.concat([data_rf, data_bpi], sort=False, ignore_index=True)
    data["error"] /= params["horizon"]
    data["error-ucb"] /= params["horizon"]
    data["epsilon"] /= params["horizon"]
    data["episodes"] = data["samples"] / params["horizon"]

    plot_bins(data, params["out"])
    plot_error(data, params["out"])
    plt.show()


def plot_bins(data, out_dir, n_bins=11):
    # plot with number of samples required for each epsilon
    colours = {"BPI-UCRL": "Reds_r", "RF-UCRL": "Greens_r"}
    for algorithm, df in data.groupby("algorithm"):
        f, ax = plt.subplots()
        data_bins = df.copy()
        data_bins = data_bins[data_bins["epsilon"].between(0.0, 1.0)]
        error_bins = np.linspace(0, 1.0, n_bins)
        for ii, ee in enumerate(error_bins[:-1]):
            ee_min = ee
            ee_max = error_bins[ii+1]-1e-8
            ee_avg = int(np.ceil(100*(ee_min+ee_max)/2.0))/100
            data_bins.loc[data_bins["epsilon"].between(ee, ee_max), "epsilon"] = ee_avg

            # Fill in empty bins with nan
            if ee_avg not in data_bins["epsilon"]:
                data_bins = data_bins.append({"algorithm": algorithm,
                                      "samples": np.nan,
                                      "epsilon": ee_avg}, ignore_index=True)

        # Stopping times plot
        sns.barplot(x="epsilon", y="episodes", data=data_bins, errcolor="red",
                    palette=colours[algorithm], ax=ax)
        ax.set(xscale="linear", yscale="log")
        # Bound plot
        x = np.linspace(0, n_bins-1, 200)
        eps = np.maximum(x / (n_bins-1), 1e-4)
        ax.plot(x, data_bins["episodes"].min()/eps**2, label=r"$\mathcal{O}(1/\varepsilon^2)$", linestyle="--")
        plt.legend(loc='upper right')
        plt.ylim([data["episodes"].min(), 1000*data["episodes"].max()])
        plt.xlabel(r"$\varepsilon$")
        plt.ylabel(r"Number of episodes")
        plt.savefig(out_dir / "error-bins-{}.png".format(algorithm))
        plt.savefig(out_dir / "error-bins-{}.pdf".format(algorithm))


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
