import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


fontsize = "32";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (15, 15),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)

#----------------------------------------------------------------------
# Plot a 1D density example

mean1, cov1 = [2, 0], [(.5, .1), (.1, .5)]
mean2, cov2 = [0, 2], [(.3, 0), (0, .3)]

a = np.random.multivariate_normal(mean1, cov1, 100)
b = np.random.multivariate_normal(mean2, cov2, 100)
print(a[1])
print("---")
print(b[2])

fig, ax = plt.subplots()
ax.set_aspect('equal')
a = plt.scatter(a[:,0], a[:,1],color='darkorange', alpha=.8, s=56, marker='o')
b = plt.scatter(b[:,0], b[:,1],color='darkgreen', alpha=.8, s=56, marker='^')
c = plt.plot(1, 1, 'g', marker='v', markersize=10.0, color='red')
circ = plt.Circle((0, 2), radius=.5, color='r', fill=False, linewidth=2.5, alpha=.7)
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('plot_classes.pdf')
circ = plt.Circle((1, 1), radius=.5, color='r', fill=False, linewidth=2.5, alpha=.7)
ax.add_artist(circ)
plt.savefig('plot_knn.pdf')
plt.show()
