from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import gym

import matplotlib.colors as colors
from envs.gridworld import GridWorld

sns.set()
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


def plot_error(data: pd.DataFrame) -> None:
    f, ax = plt.subplots()
    ax.set(xscale="log", yscale="log")
    sns.lineplot(x="samples", y="error", hue="algorithm", data=data, ax=ax)
    plt.xlabel("Number of samples")
    plt.ylabel("$|\hat{V}^*(s_0) - V^*(s_0)|$")
    # plt.title("$|\hat{V}^*(s_0) - V^*(s_0)|$ versus total number of samples")
    plt.savefig("approximation_error.pdf")
    plt.savefig("approximation_error.png")
    plt.show()


def plot_error_upper_bound(xdata: np.ndarray, error_array: np.ndarray, label: str, fignum: int) -> None:
    plt.figure(fignum)
    mean_error = error_array.mean(axis=0)
    std_error = error_array.std(axis=0)
    plt.plot(xdata, mean_error, label=label)
    plt.fill_between(xdata, mean_error-std_error, mean_error+std_error, alpha=0.25)
    plt.legend()
    plt.xlabel("number of samples")
    plt.title("$\max_a E_0(s_1, a)$")


def plot_occupancies(data: pd.DataFrame, env: gym.Env) -> None:
    if isinstance(env, GridWorld):
        plot_2d_occupancies(data, env)
    else:
        plot_1d_occupancies(data)


def plot_1d_occupancies(data: pd.DataFrame) -> None:
    sns.lineplot(x="state", y="occupancy", hue="algorithm", data=data)
    plt.yscale("log")
    # plt.title("State occupancies for {} samples".format(samples))
    plt.xlabel("State $s$")
    plt.ylabel("Number of visits $N(s)$")
    plt.savefig("occupancies.pdf")
    plt.savefig("occupancies.png")
    plt.show()


def plot_2d_occupancies(data: pd.DataFrame, env: GridWorld) -> None:
    data = data.groupby(['algorithm', 'state'], as_index=False, sort=False).mean()
    v_max = data["occupancy"].max()
    algorithms = data.algorithm.unique()

    fig = plt.figure(figsize=(10, 7))
    rows = len(algorithms) // 2
    cols = np.ceil(len(algorithms) / rows).astype(int)
    for i, algorithm in enumerate(algorithms):
        ax = fig.add_subplot(rows, cols, i+1)
        df = data[data["algorithm"] == algorithm]
        occupancies = np.zeros((env.nrows, env.ncols))
        for _, row in df.iterrows():
            occupancies[env.index2coord[row.state]] = row.occupancy
        im = ax.imshow(occupancies,
            norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=0, vmax=v_max),
            cmap=plt.cm.coolwarm)
        ax.set_title(algorithm)
    if len(algorithms) < rows * cols:
        ax = fig.add_subplot(rows, cols, rows*cols)
        ax.axis('off')
    fig.colorbar(im, ax=ax)
    plt.savefig("2d_occupancies.pdf")
    plt.savefig("2d_occupancies.png")
    plt.show()


def kullback_leibler(p: np.ndarray, q: np.ndarray) -> float:
    """
        KL between two categorical distributions
    :param p: categorical distribution
    :param q: categorical distribution
    :return: KL(p||q)
    """
    kl = 0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi > 0:
                kl += pi * np.log(pi/qi)
            else:
                kl = np.inf
    return kl


def bernoulli_kullback_leibler(p: float, q: float) -> float:
    """
        Compute the Kullback-Leibler divergence of two Bernoulli distributions.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: KL(B(p) || B(q))
    """
    kl1, kl2 = 0, np.infty
    if p > 0:
        if q > 0:
            kl1 = p*np.log(p/q)

    if q < 1:
        if p < 1:
            kl2 = (1 - p) * np.log((1 - p) / (1 - q))
        else:
            kl2 = 0
    return kl1 + kl2


def d_bernoulli_kullback_leibler_dq(p: float, q: float) -> float:
    """
        Compute the partial derivative of the Kullback-Leibler divergence of two Bernoulli distributions.

        With respect to the parameter q of the second distribution.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: dKL/dq(B(p) || B(q))
    """
    return (1 - p) / (1 - q) - p/q


def kl_upper_bound(_sum: float, count: int, threshold: float = 1, eps: float = 1e-2, lower: bool = False) -> float:
    """
        Upper Confidence Bound of the empirical mean built on the Kullback-Leibler divergence.

        The computation involves solving a small convex optimization problem using Newton Iteration

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level
    :param threshold: the maximum kl-divergence * count
    :param eps: Absolute accuracy of the Netwon Iteration
    :param lower: Whether to compute a lower-bound instead of upper-bound
    """
    if count == 0:
        return 0 if lower else 1

    mu = _sum/count
    max_div = threshold/count

    # Solve KL(mu, q) = max_div
    kl = lambda q: bernoulli_kullback_leibler(mu, q) - max_div
    d_kl = lambda q: d_bernoulli_kullback_leibler_dq(mu, q)
    a, b = (0, mu) if lower else (mu, 1)

    return newton_iteration(kl, d_kl, eps, a=a, b=b)


def newton_iteration(f: Callable, df: Callable, eps: float, x0: float = None, a: float = None, b: float = None,
                     weight: float = 0.9, display: bool = False) -> float:
    """
        Run Newton Iteration to solve f(x) = 0, with x in [a, b]
    :param f: a function R -> R
    :param df: the function derivative
    :param eps: the desired accuracy
    :param x0: an initial value
    :param a: an optional lower-bound
    :param b: an optional upper-bound
    :param weight: a weight to handle out of bounds events
    :param display: plot the function
    :return: x such that f(x) = 0
    """
    x = np.inf
    if x0 is None:
        x0 = (a + b) / 2
    if a is not None and b is not None and a == b:
        return a
    x_next = x0
    iterations = 0
    while abs(x - x_next) > eps:
        iterations += 1
        x = x_next

        if display:
            import matplotlib.pyplot as plt
            xx0 = a or x-1
            xx1 = b or x+1
            xx = np.linspace(xx0, xx1, 100)
            yy = np.array(list(map(f, xx)))
            plt.plot(xx, yy)
            plt.axvline(x=x)
            plt.show()

        f_x, df_x = f(x), df(x)
        if df_x != 0:
            x_next = x - f_x / df_x

        if a is not None and x_next < a:
            x_next = weight * a + (1 - weight) * x
        elif b is not None and x_next > b:
            x_next = weight * b + (1 - weight) * x

    if a is not None and x_next < a:
        x_next = a
    if b is not None and x_next > b:
        x_next = b

    return x_next


def max_expectation_under_constraint(f: Callable, q: np.ndarray, c: float, eps: float = 1e-2,
                                     display: bool = False) -> np.ndarray:
    """
        Solve the following constrained optimisation problem:
             max_p E_p[f]    s.t.    KL(q || p) <= c
    :param f: an array of values f(x), np.array of size n
    :param q: a discrete distribution q(x), np.array of size n
    :param c: a threshold for the KL divergence between p and q.
    :param eps: desired accuracy on the constraint
    :param display: plot the function
    :return: the argmax p*
    """
    np.seterr(all='warn')
    x_plus = np.where(q > 0)
    x_zero = np.where(q == 0)
    p_star = np.zeros(q.shape)
    lambda_, z = None, 0

    q_p = q[x_plus]
    f_p = f[x_plus]
    f_star = np.amax(f)
    theta = lambda l: q_p @ np.log(l - f_p) + np.log(q_p @ (1 / (l - f_p))) - c
    d_theta_dl = lambda l: q_p @ (1 / (l - f_p)) - (q_p @ (1 / (l - f_p)**2)) / (q_p @ (1 / (l - f_p)))
    if f_star > np.amax(f_p):
        theta_star = theta(f_star)
        if theta_star < 0:
            lambda_ = f_star
            z = 1 - np.exp(theta_star)
            p_star[x_zero] = 1.0 * (f[x_zero] == np.amax(f[x_zero]))
            p_star[x_zero] *= z / p_star[x_zero].sum()
    if lambda_ is None:
        if np.allclose(f_p, f_p[0]):
            return q
        else:
            lambda_ = newton_iteration(theta, d_theta_dl, eps, x0=f_star + 1, a=f_star, display=display)

    beta = (1 - z) / (q_p @ (1 / (lambda_ - f_p)))
    if beta == 0:
        x_uni = np.where((q > 0) & (f == f_star))
        p_star[x_uni] = (1 - z) / np.size(x_uni)
    else:
        p_star[x_plus] = beta * q_p / (lambda_ - f_p)
    return p_star


def all_argmax(x: np.ndarray) -> np.ndarray:
    """
    :param x: a set
    :return: the list of indexes of all maximums of x
    """
    m = np.amax(x)
    return np.nonzero(np.isclose(x, m))[0]


def random_argmax(x: np.ndarray) -> int:
    """
        Randomly tie-breaking arg max
    :param x: an array
    :return: a random index among the maximums
    """
    indices = all_argmax(x)
    return np.random.choice(indices)
