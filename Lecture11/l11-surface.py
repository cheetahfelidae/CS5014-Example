#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Create a dictionary to pass to matplotlib
# This is an easy way to set many parameters at once
fontsize = "24"
params = {'figure.autolayout': True,
          'legend.fontsize'  : fontsize,
          'figure.figsize'   : (8, 8),
          'axes.labelsize'   : fontsize,
          'axes.titlesize'   : fontsize,
          'xtick.labelsize'  : fontsize,
          'ytick.labelsize'  : fontsize}
plt.rcParams.update(params)

# Load the data from the space-separated txt files. 
# The inputs: 
data = np.loadtxt('moteLoc.txt')
x = data[:, 1:3]
y = np.loadtxt('labTemp.txt')

# Delete sensor 5 because it's faulty 
mask = np.ones(len(y), dtype=bool)
mask[4] = False
y = y[mask]
x = x[mask]

# Create a new figure and an axes objects for the subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, color='red', marker='o', s=140)
ax.set_xlim([-10, 50])
ax.set_ylim([-10, 50])
ax.set_zlim([15, 19])
ax.set_xticks([-10, 10, 30, 50])
ax.set_yticks([0, 20, 40])
ax.set_zticks([15, 16, 17, 18, 19])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature')

plt.savefig('plot_temperature3d.png')
# plt.show()

# Expand features to add higher degrees. Instead of doing
# this by hand like we did in the 1D case, we will use sklearn's
# PolynomialFeatures which will do the hard work for us
# Exclude bias because the linear regressor does that anyway
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# print(x_poly)

# Define the resolution of the graph. Then choose 'res' points in
# each dimension, linearly spaced between min and max
res = 30
xspace = np.linspace(-10, 50, res)
yspace = np.linspace(-10, 50, res)

# Create a grid of these two sequences
xx, yy = np.meshgrid(xspace, yspace)

# Finally, obtain a list of coordinate pairs
xy = np.c_[xx.ravel(), yy.ravel()]
xy_poly = poly.fit_transform(xy)

# Do linear regresion on original data. This will fit a plane to
# the data

linreg = LinearRegression()
linreg.fit(x, y)
new_y = linreg.predict(xy).reshape(res, res)

# ax.plot_surface(xx, yy, new_y, rstride=1, cstride=1, cmap='jet', edgecolor='none',alpha=0.6)
ax.set_xlim([-10, 50])
ax.set_ylim([-10, 50])
ax.set_zlim([15, 19])
# plt.savefig('plot_temperature_plane.png')


# Polynomial regression order 2
res = 30
xspace = np.linspace(0, 40, res)
yspace = np.linspace(0, 40, res)
xx, yy = np.meshgrid(xspace, yspace)
xy = np.c_[xx.ravel(), yy.ravel()]
xy_poly = poly.fit_transform(xy)

linreg.fit(x_poly, y)
new_y = linreg.predict(xy_poly).reshape(res, res)
# new_y[new_y<15] = 15
ax.plot_surface(xx, yy, new_y, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.6)
plt.savefig('plot_temperature_parabola.png')

exit()
