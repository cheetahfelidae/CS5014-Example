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
np.random.seed(40)

mean1, cov1 = [2, 0], [(.5, .1), (.1, .5)]
mean2, cov2 = [0, 2], [(.3, 0), (0, .3)]

a = np.random.multivariate_normal(mean1, cov1, 100)
b = np.random.multivariate_normal(mean2, cov2, 100)
        
data = np.concatenate((a,b))
datab = np.ones([1,2])*[0,2]

df = pd.DataFrame(data, columns=["x", "y"])

fig, ax = plt.subplots()
ax.set_aspect('equal')
a = plt.scatter(data[:,0], data[:,1],color='darkblue', alpha=.7, s=36)
b = plt.scatter(datab[:,0], datab[:,1],color='red', marker='v', alpha=.7, s=76)
circ = plt.Circle((0, 2), radius=.5, color='r', fill=False, linewidth=2.5, alpha=.7)
ax.add_artist(circ)
plt.legend((a, b), ('old data', 'new data'), loc='upper right')
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('plot_scatter_circle.pdf')
plt.show()
