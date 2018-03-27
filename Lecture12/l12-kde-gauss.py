import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy import stats, integrate

from mpl_toolkits.mplot3d import Axes3D

fontsize = "20";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (20, 10),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)

np.random.seed(42)

mean1, cov1 = [2, 0], [(.5, .1), (.1, .5)]
mean2, cov2 = [0, 2], [(.3, 0), (0, .3)]
a = np.random.multivariate_normal(mean1, cov1, 20)
b = np.random.multivariate_normal(mean2, cov2, 20)
data = np.concatenate((a,b))
x = data[:,0]
y = data[:,1]

bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)
support = np.linspace(-4, 6, 1000)

fig = plt.figure()

ax0 = plt.subplot2grid((2, 3), (0, 0))
ax1 = plt.subplot2grid((2, 3), (1, 0))
ax2 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2, projection='3d')

kernels = []
for x_i in x:
    kernel = stats.uniform(x_i-bandwidth/2,scale=bandwidth).pdf(support)/2
    kernels.append(kernel)
    ax0.plot(support, kernel, color="r")

density = np.sum(kernels, axis=0)
density /= integrate.trapz(density, support)

ax1.fill(support,density)

sns.rugplot(x, color=".2", linewidth=3, ax=ax0);
sns.rugplot(x, color=".2", linewidth=3, ax=ax1);

X, Y  =  np.mgrid[-2:4:.01, -2:4:.01]
Z = np.zeros_like(Y)

for l,m in data:
    print(l,m)
    for i in range(X.shape[0]):                     
        for j in range(Y.shape[1]):
            curx = X[i,j]
            cury = Y[j,j]
            dist = np.sqrt((curx-l)**2+(cury-m)**2)
            if abs(curx-l) < bandwidth/2 and abs(cury-m) < bandwidth/2:
                val = 1
            else:
                val = 0
            Z[i,j] += val

ax2.scatter(x, y, zs=-.2, zdir='z')
ax2.view_init(elev=30., azim=30)
ax2.plot_surface(X, Y, Z/30, cmap=cm.coolwarm)

ax0.axis('off')
ax0.set_xlim([-2,4])

ax1.axis('off')
ax1.set_xlim([-2,4])
plt.savefig('plot_tophat_kde.pdf',transparent=True)
plt.show()
