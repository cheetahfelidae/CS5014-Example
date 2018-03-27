import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

fontsize = "24";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (15, 5),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)

#----------------------------------------------------------------------
# Plot a 1D density example
N = 100
np.random.seed(1)
mu1 = 170
sigma1 = 20

X = np.concatenate((np.random.normal(mu1, sigma1, 1 * N),
                    np.random.normal(5, 1, int(0.7 * 0))))[:, np.newaxis]

X_plot = np.linspace(100, 250, 1000)[:, np.newaxis]
X_plot2 = np.linspace(100, 250, 1000)[:, np.newaxis]

true_dens1 = (0.3 * norm(mu1, sigma1).pdf(X_plot[:, 0]))
true_dens2 = (0.3 * norm(mu1, sigma1).pdf(X_plot2[:, 0]))
true_dens2[0:350] = 0;
true_dens2[450:999] = 0;

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens1, fc='black', alpha=0.2,
        label='Input distribution')

ax.legend(loc='upper right')

sns.rugplot(X, color='k')
plt.xlabel('x (height in cm)')
plt.ylabel('p(x)')
plt.savefig('plot_gauss.pdf')

ax.fill(X_plot[:, 0], true_dens2, fc='blue', alpha=0.2, label='Region R')
ax.legend(loc='upper right')
plt.savefig('plot_gauss2.pdf')

plt.show()
