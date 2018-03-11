import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import optimize


def cost_function(theta,X,y):
   theta.shape = (1, 3)
   m = y.size
   h = sig_fun(X.dot(theta.conj().transpose()))
   a = ((-y).T.dot(np.log(h)))
   b = (1-y).T.dot(np.log(1-h))
   J =(a - b)/m
   return J.sum()


def gradient(theta,X,y):
   theta.shape = (1, 3)
   grad = np.zeros(3)
   h = sig_fun(X.dot(theta.conj().transpose()))
   a = h - y
   l = grad.size
   for i in range(l):
      suma = a.conj().transpose().dot(X[:, i])
      grad[i] = (1.0 / m) * suma * (-1)
   theta.shape = (3,)
   return grad


def sig_fun(z):
   g=1/(1+np.exp(-z))
   return g


def learning_parameters(i, y):
    def f(theta):
        return cost_function(theta, i, y)

    def fprime(theta):
        return gradient(theta, i, y)
    theta = np.zeros(3)
    return sp.optimize.fmin_bfgs(f, theta, fprime, disp=True, maxiter=400)


def prediction(theta,X):
   m, n = X.shape
   p = np.zeros(shape=(m, 1))
   h = sig_fun(X.dot(theta.conj().transpose()))
   for i in range(0, h.shape[0]):
        if h[i] > 0.5:
            p[i, 0] = 1
        else:
            p[i, 0] = 0
   return p

features = np.loadtxt('logis.txt', delimiter=',')
X = features[:, 0:2]
y =  features[:, 2]
pos = np.where(y == 1)
neg = np.where(y == 0)
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='r')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['X', 'Y'])

m, n = X.shape
y.shape = (m, 1)
i = np.ones(shape=(m, 3))
i[:, 1:3] = X


learning_parameters(i, y)
theta = [-25.161272, 0.206233, 0.201470]

plot_x = np.array([min(i[:, 1]) - 2, max(i[:, 2]) + 2])
plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])

plt.plot(plot_x, plot_y)
plt.legend(['Decision', 'Maligenant', 'Benign'])
plt.show()
p = prediction(np.array(theta), i)
print ("Training Accuracy:",((y[np.where(p == y)].size / float(y.size)) * 100.0), "%")