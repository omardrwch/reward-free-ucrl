import matplotlib.pyplot as plt


def plot_error(xdata, error_array, label, fignum):
    plt.figure(fignum)
    mean_error = error_array.mean(axis=0)
    std_error  = error_array.std(axis=0)
    plt.plot(xdata, mean_error, label=label)
    plt.fill_between(xdata, mean_error-std_error, mean_error+std_error, alpha=0.25)
    plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("number of samples")
    plt.ylabel("$||\hat{Q}^* - Q^*||_\infty$")
    plt.title("$||\hat{Q}^* - Q^*||_\infty$ versus total number of samples")


def plot_error_upper_bound(xdata, error_array, label, fignum):
    plt.figure(fignum)
    mean_error = error_array.mean(axis=0)
    std_error  = error_array.std(axis=0)
    plt.plot(xdata, mean_error, label=label)
    plt.fill_between(xdata, mean_error-std_error, mean_error+std_error, alpha=0.25)
    plt.legend()
    plt.xlabel("number of samples")
    plt.title("$\max_a E_0(s_1, a)$")
