#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Create a dictionary to pass to matplotlib
# This is an easy way to set many parameters at once
fontsize = "30";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (16, 8),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)

np.random.seed(3)
m = 15
X = 6 * np.random.rand(m,1) - 3
y = 0.2 * X**2 + .1 * X + 2 + .3*np.random.randn(m,1)

print(X.min(), X.max())
#print(y)

# Create a new figure and an axes objects for the subplot
fig, ax = plt.subplots()

# Use a light grey grid to make reading easier, and put the grid 
# behind the display markers
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.set_axisbelow(True)
ax.set_ylim(1,4)
ax.set_xlim(-4,4)

# Draw a scatter plot of x vs y
ax.scatter(X, y, color='blue', alpha=.8, s=140, marker='^')
ax.set_xlabel('X')
ax.set_ylabel('y')

# Save as an image file
plt.savefig('plot_parabola.png')

# Linear regression
X_test = np.linspace(-3,3,100)
X_test = X_test.reshape((100,1))

new_X = X_test
linreg = LinearRegression()
linreg.fit(X,y)
new_y = linreg.predict(new_X)
plt.plot(new_X, new_y, color='orange', linewidth=3)
#plt.savefig('plot_linear.png')

# Polynomial regression order 2
X2 = np.hstack((X,X**2))
new_X2 = np.hstack((new_X,new_X**2))
linreg.fit(X2,y)
new_y = linreg.predict(new_X2)
plt.plot(new_X, new_y, color='green', linewidth=3)
#plt.savefig('plot_quadratic.png')

# Polynomial regression order 3
X3 = np.hstack((X,X**2,X**3))
new_X3 = np.hstack((new_X,new_X**2,new_X**3))
linreg.fit(X3,y)
new_y = linreg.predict(new_X3)
plt.plot(new_X, new_y, color='red', linewidth=3)
#plt.savefig('plot_cubic.png')

# Polynomial regression order 7
X7 = np.hstack((X,X**2,X**3,X**4,X**5,X**6,X**7))
new_X7 = np.hstack((new_X,new_X**2,new_X**3,new_X**4,new_X**5,new_X**6,new_X**7))
linreg.fit(X7,y)
new_y = linreg.predict(new_X7)
plt.plot(new_X, new_y, color='purple', linewidth=3)
#plt.savefig('plot_seven.png')

ax.legend(['Linear','Quadratic','Cubic','7th order'], loc='lower left')

# Display in a window
plt.show()
