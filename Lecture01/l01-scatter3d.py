# Create a 3D scatter plot of the data from the catheter dataset
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

# It is useful to print them, just to make sure we got it right
print(x)
print(y)

# Create a new figure and an axes objects for the subplot
# We only have one plot here, but it's helpful to be consistent
# Note that we specify a 3D projection here
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D scatter plot (x1,x2,y), with y being the height
ax.scatter(x[:, 0], x[:, 1], y, color='red', marker='o', s=140)

# Now do a 2D scatter plot of x1 vs x2 only. This represents
# the projection of the datapoints onto the x1-x2 plane
ax.scatter(x[:, 0], x[:, 1], zs=0.01, color='gray', marker='o', s=140)

# Label the axes to make the plot understandable
ax.set_xlabel('x1 = Height (in)')
ax.set_ylabel('x2 = Weight (lbs)')
ax.set_zlabel('y = Cath length (cm)')

# One of the axes is a bit too large, so set the limits manually
ax.set_zlim(0, 60)

# Save as an image file
plt.savefig('plot_scatter3D.png')

# Display in a window
plt.show()
