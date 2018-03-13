# Create a 2D scatter plot of the data from the catheter dataset
#
# Kasim Terzic (kt54) Jan 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
fig = plt.figure()
ax = fig.add_subplot(111)

# Use a light grey grid to make reading easier, and put the grid 
# behind the display markers
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.set_axisbelow(True)

# Draw a scatter plot of the first column of x vs second column.
ax.scatter(x[:, 0], x[:, 1], color='red', alpha=.8, s=140, marker='^')
ax.set_xlabel('Height (in)')
ax.set_ylabel('Weight (lbs)')

# Save as an image file
plt.savefig('plot_scatter2D.png')

# Display in a window
plt.show()
