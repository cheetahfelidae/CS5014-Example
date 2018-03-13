#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Create a dictionary to pass to matplotlib
# This is an easy way to set many parameters at once
fontsize = "30"
params = {'figure.autolayout': True,
          'legend.fontsize'  : fontsize,
          'figure.figsize'   : (16, 8),
          'axes.labelsize'   : fontsize,
          'axes.titlesize'   : fontsize,
          'xtick.labelsize'  : fontsize,
          'ytick.labelsize'  : fontsize}
plt.rcParams.update(params)

np.random.seed(3)
m = 40
width = 10
height = 5
X = width * np.random.rand(m, 1) - width / 2
y = height / 2 * np.sin(X) + .3 * np.random.randn(m, 1)

# Create a new figure and an axes objects for the subplot
fig, ax = plt.subplots()

# Use a light grey grid to make reading easier, and put the grid 
# behind the display markers
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.set_axisbelow(True)
# ax.plot((-width/4,-10),(-width/4,10),c='k')
ax.vlines([-width / 4, width / 4], -4, 4, colors='k', linestyles='dashed')
ax.set_ylim(-4, 4)

# Draw a scatter plot of x vs y
ax.scatter(X, y, color='blue', alpha=.8, s=140, marker='^')
ax.set_xlabel('X')
ax.set_ylabel('y')

# Save as an image file
plt.savefig('plot_sinewave.png')

# Linear regression
X_test = np.linspace(-width / 2, width / 2, 300)
X_test = X_test.reshape((300, 1))

new_X = X_test

knot1_train = np.clip(X + width / 4, 0, None)
knot2_train = np.clip(X - width / 4, 0, None)
knot1_test = np.clip(X_test + width / 4, 0, None)
knot2_test = np.clip(X_test - width / 4, 0, None)
X1 = np.hstack((X, knot1_train, knot2_train))
new_X = np.hstack((X_test, knot1_test, knot2_test))
linreg = LinearRegression()
linreg.fit(X1, y)
new_y = linreg.predict(new_X)
plt.plot(X_test, new_y, color='orange', linewidth=3)
# ax.legend(['Piecewise Linear Fit'], loc='lower left')
# plt.savefig('plot_piecewise_linear.png')

# Polynomial regression order 2
knot1_train = np.clip(X + width / 4, 0, None)
knot2_train = np.clip(X - width / 4, 0, None)
knot1_test = np.clip(X_test + width / 4, 0, None)
knot2_test = np.clip(X_test - width / 4, 0, None)
X2 = np.hstack((X, X ** 2, knot1_train ** 2, knot2_train ** 2))
new_X = np.hstack((X_test, X_test ** 2, knot1_test ** 2, knot2_test ** 2))
linreg = LinearRegression()
linreg.fit(X2, y)
new_y = linreg.predict(new_X)
plt.plot(X_test, new_y, color='green', linewidth=3)
# ax.legend(['Piecewise Quadratic Fit'], loc='lower left')
# plt.savefig('plot_piecewise_quadratic.png')


# Polynomial regression order 3
knot1_train = np.clip(X + width / 4, 0, None)
knot2_train = np.clip(X - width / 4, 0, None)
knot1_test = np.clip(X_test + width / 4, 0, None)
knot2_test = np.clip(X_test - width / 4, 0, None)
X2 = np.hstack((X, X ** 2, X ** 3, knot1_train ** 3, knot2_train ** 3))
new_X = np.hstack((X_test, X_test ** 2, X_test ** 3, knot1_test ** 3, knot2_test ** 3))
linreg = LinearRegression()
linreg.fit(X2, y)
new_y = linreg.predict(new_X)
plt.plot(X_test, new_y, color='red', linewidth=3)
# ax.legend(['Piecewise Cubic Fit'], loc='lower left')
# plt.savefig('plot_piecewise_cubic.png')

ax.legend(['Linear', 'Quadratic', 'Cubic'], loc='lower left')

# Display in a window
plt.show()
