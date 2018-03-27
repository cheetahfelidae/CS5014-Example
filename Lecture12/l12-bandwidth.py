# Author: Jake Vanderplas <jakevdp@cs.washington.edu>
# Adapted for CS5014 by kt54@st-andrews.ac.uk
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import seaborn

fontsize = 20;
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (15, 10),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)


#----------------------------------------------------------------------
# Plot a 1D density example
N = 100
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

fig, axe = plt.subplots(2,3,sharex=True, sharey=True)
ax = axe[0][0]

for axi in axe.ravel():
    axi.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
        label='input distribution')

kernels= ['tophat', 'gaussian']
bws = [0.1, .7, 3]

for j in range(2):
    for i in range(3):
        kde = KernelDensity(kernel=kernels[j], bandwidth=bws[i]).fit(X)
        log_dens = kde.score_samples(X_plot)
        axe[j,i].plot(X_plot[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernels[j], sharex=axe[j,0]))

for i in range(3):
    axe[0][i].set_title("h = {}".format(bws[i]))


ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.savefig('plot_bw.pdf')
plt.show()
