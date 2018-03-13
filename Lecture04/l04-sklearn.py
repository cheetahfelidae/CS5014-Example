# This is an example of linear regression, but using sklearn's algorithm
#
# Kasim Terzic (kt54) Feb 2018

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data from the space-separated txt file. 
# This will create a 2D numpy array
# data = np.loadtxt('l04-data.txt')
data = np.loadtxt('/Users/cheetah/Sites/CS5014-Example/Lecture04/l01-data.txt')

# Extract column 2 (X) and 3 (Y). We do not need to manually 
# add a vector of ones, because sklearn will do this automatically
x = data[:, 1]
y = data[:, 2]

# use sklearn to perform linear regression
# this will use a linear solver (covered in W3)
linreg = LinearRegression()
linreg.fit(x, y)
y_hat = linreg.predict(x)
print('MSE = ', mean_squared_error(y, y_hat))
