#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a dictionary to pass to matplotlib
# This is an easy way to set many parameters at once
fontsize = "30"
params = {'figure.autolayout': True,
          'legend.fontsize'  : fontsize,
          'figure.figsize'   : (8, 8),
          'axes.labelsize'   : fontsize,
          'axes.titlesize'   : fontsize,
          'xtick.labelsize'  : fontsize,
          'ytick.labelsize'  : fontsize}
plt.rcParams.update(params)

np.random.seed(1)
m = 10
X = 3 * np.random.rand(m, 1) - 3
X2 = X + 3

X_all = np.vstack((X, X2))

y = 0.4 * X + X ** 2 + .2 * np.random.randn(m, 1)
y2 = 0.4 * X2 + X2 ** 2 + .2 * np.random.randn(m, 1)

y_all = 0.4 * X_all + X_all ** 2 + .2 * np.random.randn(m * 2, 1)

# Create a new figure and an axes objects for the subplot
fig, ax = plt.subplots()

# Use a light grey grid to make reading easier, and put the grid 
# behind the display markers
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.set_axisbelow(True)
# ax.set_ylim(1,4)
# ax.set_xlim(-4,4)

# Draw a scatter plot of x vs y
ax.scatter(X_all, y_all, color='blue', alpha=.8, s=140, marker='^')
ax.set_xlabel('X')
ax.set_ylabel('y')
plt.savefig('plot_test1-all.png')

fig, ax = plt.subplots()
# Use a light grey grid to make reading easier, and put the grid 
# behind the display markers
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.set_axisbelow(True)
# ax.set_ylim(1,4)
# ax.set_xlim(-4,4)

# Draw a scatter plot of x vs y
ax.scatter(X, y, color='green', alpha=.8, s=140, marker='^')
ax.set_xlabel('X')
ax.set_ylabel('y')

# Linear regression
X_test = np.linspace(-3, 3, 100)
X_test = X_test.reshape((100, 1))

new_X = X_test
linreg = LinearRegression()
linreg.fit(X, y)
new_y = linreg.predict(new_X)
plt.plot(new_X, new_y, color='orange', linewidth=3)
plt.savefig('plot_test1-train.png')

# Calculate mean squared error
pred_y = linreg.predict(X)
print(mean_squared_error(y, pred_y))
plt.show()

## now test set
m = 10

fig, ax = plt.subplots()
# Draw a scatter plot of x vs y
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.set_axisbelow(True)
ax.scatter(X2, y2, color='red', alpha=.8, s=140, marker='^')
ax.set_xlabel('X')
ax.set_ylabel('y')

# Linear regression
plt.plot(new_X, new_y, color='orange', linewidth=3)
plt.savefig('plot_test1-test.png')

# Calculate mean squared error
pred_y = linreg.predict(X2)
plt.show()
print(mean_squared_error(y2, pred_y))
