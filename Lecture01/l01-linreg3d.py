# Create a 3D plot of the catheter dataset, showing the
# length predicted by a linear model in 3D
#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Load the data from the space-separated txt file. 
# This will create a 2D numpy array
data = np.loadtxt('/Users/cheetah/Sites/CS5014-Example/Lecture01/l01-data.txt')

# Extract columns 1 and 2 (X) and 3 (Y)
x = data[:, 1:3]
y = data[:, 3]

# Create a new figure and an axes objects for the subplot
# We only have one plot here, but it's helpful to be consistent
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# We will sample both X1 and X2 dimensions and evaluate the model
# at each X1,X2 coordinate in turn. Our model is simply the Y value
# of the nearest neighbour

# Define the resolution of the graph. Then choose 'res' points in
# each dimension, linearly spaced between min and max
res = 30
xspace = np.linspace(0, 80, res)
yspace = np.linspace(0, 100, res)

# Create a grid of these two sequences
xx, yy = np.meshgrid(xspace, yspace)

# Finally, obtain a list of coordinate pairs
xy = np.c_[xx.ravel(), yy.ravel()]

# Create an output array of right dimensions. The values don't matter
t = xy[:, 0].reshape(res, res)

tt = t.reshape(-1);  # Convert to 1D to make the loop easier...

# For each X1,X2 pair, calculate the value using a linear model
# The actual parameters here were rather unscientifically made up
for index, val in enumerate(tt):
    tt[index] = 32. * index / (res * res) + 18.

tt = tt.reshape(res, res)  # Convert to 2D again

# Now plot these values as a surface. tt represents the predicted Y value 
# of each X1 and X2 pair
# We also added a colourmap, so we can match this graph to the 2D one 
ax.plot_surface(xx, yy, tt, rstride=1, cstride=1, cmap='jet', edgecolor='none', alpha=0.6)

# then add the original datapoints on top as reference
ax.scatter(x[:, 0], x[:, 1], y, color='k', marker='o', s=140)

# change the viewing angle so we can actually see what's happening 
ax.view_init(elev=10., azim=-20)

# Label the axes to make the plot understandable
ax.set_xlabel('x1 = Height (in)')
ax.set_ylabel('x2 = Weight (lbs)')
ax.set_zlabel('y = Cath length (cm)')

# change the numbering on the X1 and X2 axes. At the desired viewing angle,
# numbers were jumbled together. This cosmetic change makes them readable again
ax.set_xticks([0, 40, 80])
ax.set_yticks([-20, 20, 60, 100])

# Save as an image file
plt.savefig('plot_linreg3d.png')

# Display in a window
plt.show()
