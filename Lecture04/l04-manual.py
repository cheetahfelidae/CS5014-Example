# Kasim Terzic (kt54) Feb 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Lecture04.l04_utils import *

# Load the data from the space-separated txt file. 
# This will create a 2D numpy array
# data = np.loadtxt('l04-data.txt')
data = np.loadtxt('/Users/cheetah/Sites/CS5014-Example/Lecture04/l01-data.txt')
factor = np.max(data)

# Extract column 2 (X) and 3 (Y). Add a vector of ones as X_0
x = data[:, 1] / factor
y = data[:, 2] / factor
x = np.c_[np.ones_like(x), x]

# Pick some parameters for our model
theta_start = np.array([0, 0])

theta_new, loss = gradientDescent(x, y, 0.1, theta_start, 1e-8)
print(theta_new, loss)

print(meanSquaredErrorLoss(y, f(x, theta_start)))
plotModel(x, y, f(x, theta_start), title='Random parameters')

print(meanSquaredErrorLoss(y, f(x, theta_new)))
plotModel(x, y, f(x, theta_new), title='After gradient descent')

plotLossFunction(x, y, title='Loss function')
plt.savefig('l04-plot.png')
plt.show()
