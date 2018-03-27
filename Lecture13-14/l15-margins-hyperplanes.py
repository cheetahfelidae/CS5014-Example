import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression

fontsize = "30";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (12, 8),
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize}
plt.rcParams.update(params)
    
# we create 40 separable points
np.random.seed(0)
npc = 30
X = np.r_[np.random.randn(npc, 2) - [1, 1], np.random.randn(npc, 2) + [1,1]]
y = [0] * npc + [1] * npc

X_train = X
y_train = y

n_sample = len(X)

x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()
logreg = LogisticRegression()
logreg.fit(X,y)
plt.figure(0)
plt.clf()
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = logreg.decision_function(np.c_[XX.ravel(), YY.ravel()])
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap="binary", #plt.cm.Paired,
            edgecolor='k', s=40)
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap="rainbow", alpha=.1, shading="flat", edgecolors="None")
plt.contour(XX, YY, Z, colors=['k'],
            linestyles=['-'], levels=[0])
plt.title("Logistic Regression")
plt.savefig('logreg_train.png')

# Now loop through some SVM kernels
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num+1)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap="binary", #plt.cm.Paired,
                edgecolor='k', s=40)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                facecolors='none', zorder=10, edgecolors='r')

    # Circle out the test data
    #plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
    #            zorder=10, edgecolor='k')

    plt.axis('tight')
    #x_min = X[:, 0].min()
    #x_max = X[:, 0].max()
    #y_min = X[:, 1].min()
    #y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap="rainbow", alpha=.1, shading="flat", edgecolors="None")
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-1, 0, 1])

    title = 'SVM with '+kernel+' kernel';
    plt.title(title)
    plt.savefig(title)

#plt.show()
