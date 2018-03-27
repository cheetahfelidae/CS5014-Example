# Example histogram plot
# I ripped so many things off to make this, I forgot most of them

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from mpl_toolkits.mplot3d import Axes3D

fontsize = "32";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (15, 10),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)

def sph2cart(r, theta, phi):
    '''spherical to cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def ravzip(*itr):
    '''flatten and zip arrays'''
    return zip(*map(np.ravel, itr))


def getDistances(view):
    distances  = []
    a = np.array((xpos, ypos, dz))
    for i in range(len(xpos)):
        distance = (a[0, i] - view[0])**2 + (a[1, i] - view[1])**2 + (a[2, i] - view[2])**2
        distances.append(np.sqrt(distance))
    return distances

#----------------------------------------------------------------------
# Plot a 1D density example

mean1, cov1 = [2, 0], [(.5, .1), (.1, .5)]
mean2, cov2 = [0, 2], [(.3, 0), (0, .3)]

a = np.random.multivariate_normal(mean1, cov1, 1000)
b = np.random.multivariate_normal(mean2, cov2, 1000)
c = np.random.uniform(-2,5,1000);
d = np.random.uniform(-3,4,1000);
cd = np.vstack((c,d)).T
        
data = np.concatenate((a,b))

x = data[:,0]
y = data[:,1]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(x, y, bins=12, range=[[-2, 5], [-3, 4]])

# Construct arrays for the anchor positions of the 16 bars.
# Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
# ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
# with indexing='ij'.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.view_init(elev=10., azim=30)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#aaaaff', zsort='max', alpha=1)
###ax1.bar3d(X.ravel(), Y.ravel(), np.zeros(X.size), dx, dy, Zg.ravel(), '0.85')


plt.savefig('plot_hist3d.pdf', transparent=True)
plt.show()
