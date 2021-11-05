import matplotlib


# editor must be jupyterlab
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy.core.numeric import ones


def draw(x1, x2):
    # to draw a line
    ln = plt.plot(x1, x2)


def sigmoid(score):
    # calculate the score of every points
    return 1/(1 + np.exp(-score))


np.random.seed(0)
# number of points
n_pts = 100
bias = np.ones(n_pts)
# normal distribution
random_x1_values = np.random.normal(10, 2, n_pts)
random_x2_values = np.random.normal(12, 2, n_pts)
# creating a chart for points
top_region = np.array([random_x1_values, random_x2_values, bias]).transpose
bottom_region = np.array(
    [np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
# blend to matrices vertically
all_points = np.vstack((top_region, bottom_region))
w1 = -0.2
w2 = -0.35
b = 3.5
line_parameters = np.matrix([w1, w2, b]).T
linear_combination = all_points * line_parameters
probabilities = sigmoid(linear_combination)
x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
x2 = -b/w2 + x1 * (-w1/w2)
_, ax = plt.subplot(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
draw(x1, x2)
plt.show()
