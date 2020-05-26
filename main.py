import numpy as np
import pandas as pd

from agents.base_agent import experiment
from agents.mb_qvi import MB_QVI
from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.bpi_ucrl import BPI_UCRL
from envs.chain import Chain
from envs.doublechain import DoubleChain
from utils import plot_error, plot_error_upper_bound

np.random.seed(1253)

# Create parameters
params = {}
params["env"]            = DoubleChain(31, 0.25)
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

    # Run RF_UCRL with clipping
    params["clip"] = True
    results = experiment(RF_UCRL, params)
    data = data.append(results.assign(algorithm="RF-UCRL with clip"), sort=False)

    # Run RF_UCRL without clipping
    params["clip"] = False
    results = experiment(RF_UCRL, params)
    data = data.append(results.assign(algorithm="RF-UCRL without clip"), sort=False)

    # Run BPI_UCRL
    results = experiment(BPI_UCRL, params)
    data = data.append(results, sort=False)

    data.to_csv('data.csv')
    plot_error(data)


def show_occupations(samples=1000):
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.colors as colors
    del params["clip"]
    agents = {
        "RandomBaseline": RandomBaseline(**params),
        "MB_QVI": MB_QVI(**params),
        "RF_UCRL with clip": RF_UCRL(**params, clip=True),
        "RF_UCRL without clip": RF_UCRL(**params, clip=False),
    }

    plt.figure("occupations")
    for name, agent in agents.items():
        agent.run(samples)
        data = agent.N_sa.sum(axis=1) 
        plt.semilogy(data, label=name)
    plt.legend()
    plt.show()
    
    # fig = plt.figure(1, (6, 6))
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                  nrows_ncols=(4, 1),
    #                  direction="row",
    #                  axes_pad=1,
    #                  add_all=True,
    #                  label_mode="1",
    #                  share_all=True,
    #                  cbar_location="right",
    #                  cbar_mode="single",
    #                  cbar_size="1%",
    #                  cbar_pad="3%",
    # )
    # for ax, (name, agent) in zip(grid, agents.items()):
    #     agent.run(samples)
    #     occupations = agent.N_sa.sum(axis=1, keepdims=True).T
    #     vmax = samples/10
    #     im = ax.imshow(occupations, origin="lower", interpolation="nearest",
    #                    norm=colors.SymLogNorm(linthresh=1, linscale=1, vmax=vmax),
    #                    cmap=plt.cm.coolwarm)
    #     ax.cax.colorbar(im)
    #     ax.set_title(name)
    # plt.show()


if __name__=="__main__":
    estimation_error()
    show_occupations()
