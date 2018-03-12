#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Create a dictionary to pass to matplotlib
# This is an easy way to set many parameters at once
fontsize = "30";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (24, 8),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)

X = np.linspace(-4,4,100)
inputs = [X, X, X]

H = [np.ones_like(X), X, X**2]
Hdesc = ['$h_1(X) = 1$','$h_2(X) = X$','$h_3(X) = X^2$']
Colours = ['blue', 'red', 'green']

XLims = [[-1, 4]] * 3
YLims = [[-1, 4]] * 3

for j, (x, y, xlim, ylim,c,desc) in enumerate(zip(inputs, H, XLims, YLims, Colours,Hdesc)):
    ax = plt.subplot(1, 3, j + 1)
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.plot(x, y, color=c, alpha=.8, linewidth=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(desc)
    ax.set_xlabel('X')

plt.savefig('plot_bases.png')
# Display in a window
plt.show()

X = np.linspace(-4,4,100)
inputs = [X, X, X]

H = [X, np.clip(X+2.5,0,None), np.clip(X-2.5,0,None)]
Hdesc = ['$h_2(X) = X$','$h_3(X) = (X-\\xi_1)_+$','$h_4(X) = (X-\\xi_2)_+$']
Colours = ['red', 'purple', 'orange']

XLims = [[-4, 4]] * 3
YLims = [[-1, 4]] * 3

for j, (x, y, xlim, ylim,c,desc) in enumerate(zip(inputs, H, XLims, YLims, Colours,Hdesc)):
    ax = plt.subplot(1, 3, j + 1)
    ax.grid(color='lightgray', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    ax.plot(x, y, color=c, alpha=.8, linewidth=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(desc)
    ax.set_xlabel('X')

plt.savefig('plot_bases_piecewise.png')
# Display in a window
plt.show()


