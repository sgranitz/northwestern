## Week 9: Probability Density Functions
# Consider the probability density function f(x) = (3/26)x2 on [1, 3]. 
# On the same interval, consider the functions g(x) = (3/26)x3 and 
# h(x) = (x – 30/13)(3/26)x3, which when integrated over the interval [1, 3] 
# represent the mean and variance, respectively. Using Python, verify that f(x) 
# is a probability density function, that the mean is 30/13, the variance is 
# approximately 0.2592 and determine the standard deviation.  
# Also, use Python to graph these three functions together (use different colors for each) 
# and indicate the mean and variance on the x-axis.

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def f(x):
    y = (3 / 26) * x ** 2
    return (y)

def g(x):
    y = (3 / 26) * x ** 3
    return (y)

def h(x):
    y = (x - 30 / 13) * (3 / 26) * x ** 3
    return (y)

x = np.arange(1, 3.05, 0.05)

res = sp.integrate.quad(f, 1, 3)
print('Is f(x) prob density?', 1 == res[0])
# True

mean = sp.integrate.quad(g, 1, 3)
print('Is mean 30/13?', round(30/13, 4) == round(mean[0], 4))
# True

var = sp.integrate.quad(h, 1, 3)
print('Is variance 0.2592?', 0.2592 == round(var[0], 4))
# True 

plt.plot(x, f(x), label = 'f(x)')
plt.plot(x, g(x), label = 'g(x)')
plt.plot(x, h(x), label = 'h(x)')
plt.plot(mean[0], 0, 'rs', label = 'mean')
plt.plot(var[0], 0, 'bs', label = 'var')
plt.text(1, 2.5, ['Mean =', round(mean[0], 4)])
plt.text(1, 2, ['Var =', round(var[0], 4)])
plt.legend()
plt.show()
