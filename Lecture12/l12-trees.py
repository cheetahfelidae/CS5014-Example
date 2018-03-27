import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


fontsize = "28";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (20, 5),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)

mean1, cov1 = [2, 0], [(.5, .1), (.1, .5)]
mean2, cov2 = [0, 2], [(.3, 0), (0, .3)]

a = np.random.multivariate_normal(mean1, cov1, 100)
b = np.random.multivariate_normal(mean2, cov2, 100)
        
data = np.concatenate((a,b))

df = pd.DataFrame(data, columns=["x", "y"])

fig, ax = plt.subplots(1,4)
for i in range(4):
    ax[i].set_aspect('equal')
    a = ax[i].scatter(data[:,0], data[:,1],color='darkblue', alpha=.2)

for i in range(-1,5):
    for j in range(-2,4):
        ax[0].plot([i,i],  [-3,4], 'r-', lw=1)
        ax[0].plot([-2,5], [j,j], 'r-', lw=1)
ax[0].set_title('Histogram')
ax[0].axis('off')

ax[1].plot([-2,5], [1,1], 'r-', lw=2)
ax[1].plot([2,2],  [1,4], 'r-', lw=2)
ax[1].plot([0,0],  [-3,1], 'r-', lw=2)
ax[1].plot([0,5], [-1,-1], 'r-', lw=2)
ax[1].plot([3,3],  [-1,1], 'r-', lw=2)
ax[1].plot([-2,2],  [3,3], 'r-', lw=2)
ax[1].set_title('Decision Tree')
ax[1].axis('off')

c = ax[2].plot(0, 2, 'g', marker='v', markersize=10.0, color='red')
circ = plt.Rectangle((-.5, 1.5), 1, 1, color='r', fill=False, linewidth=2, alpha=1)
ax[2].add_artist(circ)
ax[2].set_title('Tophat Kernel')
ax[2].axis('off')

b = ax[3].plot(0, 2, 'g', marker='v', markersize=10.0, color='red')
circ = plt.Circle((0, 2), radius=.8, color='r', fill=False, linewidth=2, alpha=1)
ax[3].add_artist(circ)
ax[3].set_title('kNN')
ax[3].axis('off')

for i in range(4):
    ax[i].set_xlim([-2,5])
    ax[i].set_ylim([-3,4])

plt.savefig('plot_trees.pdf')
plt.show()
