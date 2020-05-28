import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
horizon = 10

# extract data
data = pd.read_csv('rf_sample_complexity.csv')
data["error"] /= horizon
data["error-ucb"] /= horizon

# plot with number of samples required for each epsilon
data_bins  = data.copy()
data_bins = data_bins[data_bins["error-ucb"].between(0.0, 1.0)]
error_bins = np.linspace(0.1, 1.0, 10)
for ii, ee in enumerate(error_bins[:-1]):
    ee_min = ee 
    ee_max = error_bins[ii+1]-1e-8
    data_bins.loc[ data_bins["error-ucb"].between(ee, ee_max), "error-ucb" ] = int(100*(ee_min+ee_max)/2.0)/100

# plt.figure("stopping times")
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
f, ax = plt.subplots()
ax.set(xscale="linear", yscale="log")
sns.barplot(x="error-ucb", y="samples", data=data_bins, palette="Blues_d", errcolor="red")
plt.xlabel("$\epsilon$")

# plot error and error UCB
def plot_error(data, fignum=1):
    plt.figure(fignum)
    f, ax = plt.subplots()
    ax.set(xscale="log", yscale="log")
    sns.lineplot(x="samples", y="error", data=data, ax=ax, label="Error (empirical)")
    sns.lineplot(x="samples", y="error-ucb", data=data, ax=ax, label="Error (UCB)")
    plt.xlabel("Number of samples")
    plt.ylabel("Error (normalized by H)")
    plt.title("$\max_a E_0(s_1, a) / H$")

plot_error(data)
plt.show()
