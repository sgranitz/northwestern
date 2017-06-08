# 1. A famous researcher observed that chimpanzees hunt and eat meat as part of their regular diet. 
# Sometimes chimpanzees hunt alone, while other times they form hunting parties. The following table 
# summarizes research on chimpanzee hunting parties, giving the size of the hunting party and the percentage 
# of successful hunts.  Use Python to graph the data and find the least squares line. Then use the equation 
# to predict the percentage of successful hunts, and the rate that percentage is changing, if there are 20 
# chimpanzees in a hunting party.
  
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15, 16])
y = np.array([15, 25, 23, 39, 35, 53, 40, 57, 60, 58, 70, 70, 73, 70, 70, 77])
A = np.vstack([x, np.ones(len(x))]).T
print(A)

m, c = np.linalg.lstsq(A, y)[0]
print(m, c)

x2 = np.append([x], [20])
y2 = np.append([y], [m * 20 + c])
plt.plot(x, y, 'o', label = 'Data', markersize = 6)
plt.plot(x2, m * x2 + c, 'b', label = 'Fitted')
plt.plot(20, m * 20 + c, 'rs')
plt.legend()
plt.show()

print('Expected Percentage with 20 Chimps:', m * 20 + c)
print('Rate of change:', m)

# 2. One gram of soybean meal provides at least 2.5 units of vitamins and 5 calories.  
# One gram of meat byproducts provides at least 4.5 units of vitamins and 3 calories.  
# One gram of grain provides at least 5 units of vitamins and 10 calories.  If a gram of 
# soybean costs 6 cents, a gram of meat byproducts costs 7 cents, and a gram of grain costs 8 cents, 
# use Python to determine what mixture of the three ingredients will provide at least 48 units of vitamins 
# and 54 calories per serving at a minimum cost?  What will be the minimum cost?

from scipy.optimize import linprog as lp
import numpy as np

# minimize: 6x + 7y + 8z
# subject to:
#   2.5x + 4.5y + 5z >= 48
#   5x + 3y + 10z >= 54
#   x, y, z  >= 0

A = np.array([[-2.5, -4.5, -5], [-5, -3, -10]])
b = np.array([-48, -54])
buy = lp(np.array([6, 7, 8]), A, b)

print('Buy', buy.x[0], 'grams soybean,', 
      buy.x[1], 'grams meat byproducts, and',
      buy.x[2], 'grams grain.')
print('Minimum per serving cost = $', 
      sum(buy.x * np.array([6, 7, 8])) / 100)

# 3. A new test has been developed to detect a particular type of cancer.  The test must be evaluated before it is put 
# into use.  A medical researcher selects a random sample of 1,000 adults and finds (by other means) that 4% have this 
# type of cancer. Each of the 1,000 adults is given the new test, and it is found that the test indicates cancer in 99% 
# of those who have it and in 1% of those who do not.  
# a) Based on these results, what is the probability of a randomly chosen person having cancer given that the test indicates cancer? 
# b) What is the probability of a person having cancer given that the test does not indicate cancer?

has_cancer = 0.04
true_pos = 0.99
false_pos = 0.01

a = has_cancer * true_pos
b = (1 - has_cancer) * false_pos
prob = a / (a + b)
print("a) Probability that person has cancer given that the test indicates cancer:",
      round(prob * 100, 3), "%.")

a = has_cancer * (1 - true_pos) # False neg
b = (1 - has_cancer) * (1 - false_pos) # True neg
prob = a / (a + b)
print("b) Probability that person has cancer given that the test doesn't indicates cancer:",
      round(prob * 100, 3), "%.")

# 4. The following is a graph of a third-degree polynomial with leading coefficient 1.  Determine the function depicted 
# in the graph.  Using Python, recreate the graph of the original function, 𝑓(𝑥), as well as the graph of its first and 
# second derivatives.

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    y = (x + 3) * (x + 1) * (x - 2)
    return (y)

def g(x):
    #f(x) = x^3 + 2x^2 - 5x - 6
    y = 3 * x ** 2 + 4 * x - 5
    return(y)

def h(x):
    y = 6 * x + 4
    return(y)

x = np.arange(-4, 4.5, 0.5)

plt.plot(x, f(x), 'b', label = 'y = f(x)')
plt.plot(x, g(x), 'b-.', label = "y = f'(x)")
plt.plot(x, h(x), 'b--', label = "y = f''(x)")
plt.plot( 1, f( 1), 'ys')
plt.plot(-2, f(-2), 'ys')
plt.plot( 2, f( 2), 'rs')
plt.plot(-1, f(-1), 'rs')
plt.plot(-3, f(-3), 'rs')
plt.plot(x, x * 0, 'g')
plt.legend()
plt.show()

# 5. For a certain drug, the rate of reaction in appropriate units is given by 𝑅'𝑡=4/(𝑡+1) + 3/sqrt(𝑡+1) where 𝑡 is time (in hours) 
# after the drug is administered. Calculate the total reaction to the drug from 𝑡=1 to 𝑡=12.

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(t):
    y = 4 * np.log(1 + t) + 6 * np.sqrt(1 + t)
    return (y)

def g(t):
    y = 4 / (t + 1) + 3 / (np.sqrt(1 + t))
    return (y)

t = np.arange(1, 12.5, 0.5)
plt.plot(t, f(t), 'b', label = 'R(t)')
plt.plot(t, g(t), 'r', label = "R'(t)")
plt.legend()
plt.show()

total = f(12) - f(1)
print(total)

# Confirm
func = lambda t: 4 / (t + 1) + 3 / (np.sqrt(1 + t))
total = integrate.quad(func, 1, 12)
print(total[0])
print("Total reaction from t=1 to t=12 is", 
      round(total[0], 4))

# 6. The nationwide attendance per day for a certain summer blockbuster can be approximated using the equation 
# 𝐴(𝑡)=13𝑡^2𝑒^-t, where A is the attendance per day in thousands of people and t is the number of months since the 
# release of the film. Find and interpret the rate of change of the daily attendance after 4 months and interpret 
# the result.

import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt

def A(t):
    y = 13 * t**2 * np.exp(-t)
    return (y)

h = 1e-5
a = (A(4+h) - A(4)) / h
print(a)
print(A(np.arange(0, 6, 1)))

t = np.arange(0, 10, 1)
plt.plot(t, A(t), 'b', label = 'A(t)')
plt.plot(t, derivative(A, t, dx = 1e-5), 'r--', label = "A'(t)")
plt.legend()
plt.show()

# Confirm
der = derivative(A, 4, dx = 1e-5)
print(der)
print("Rate of change at t=4", der)

# 7. The population of mathematicians in the eastern part of Evanston is given by the formula 𝑃(𝑡)=(𝑡^2+100) * ln(𝑡+2), 
# where t represents the time in years since 2000. Using Python, find the rate of change of this population in 2006.

import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt

def P(t):
    y = (t**2 + 100) * np.log(t + 2)
    return (y)

h = 1e-5
a = (P(6 + h) - P(6)) / h
print(a)

t = np.arange(0, 10, 1)
plt.plot(t, P(t), 'b', label = 'P(t)')
plt.plot(t, derivative(P, t, dx = 1e-5), 'r--', label = "P'(t)")
plt.legend()
plt.show()

# Confirm
der = derivative(P, 6, dx = 1e-5)
print(der)
print("Rate of change at t=2006", der)

# 8. The rate of change in a person's body temperature, with respect to the dosage of x milligrams of a certain diuretic, 
# is given by 𝐷′(x) = 2 / (x+6). One milligram raises the temperature 2.2°C. Find the function giving the total 
# temperature change. If someone takes 5 milligrams of this diuretic what will their body temperature be, assuming they 
# start off at a normal temperature of 37oC?

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    y = 2 * np.log(x + 6)
    return (y)

def g(x):
    y = 2 / (x + 6)
    return (y)

temp = 37
C = 2.2 - f(1)

t = np.arange(0, 5.5, 0.5)
plt.plot(t, f(t) + C, 'b', label = 'D(x)')
plt.plot(t, g(t), 'r--', label = "D'(x)")
plt.legend()
plt.show()

new_temp = f(5) + C + temp
print(new_temp)
print("Temp after taking 5 milligrams=", new_temp)

# 9. The following function represents the life of a lightbulb in years based on average consumer usage. 
# Show 𝑓(𝑥) is a probability density function on [0,∞).   
# 𝑓(𝑥) = x^3/12, if 0 <= x <= 2
# 𝑓(𝑥) = 16/x^4, if x > 2
# Determine the probability a lightbulb will last between 1 and 5 years.

import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(0, 2.1, 0.1)
x2 = np.arange(2.1, 20, 0.1)

def f(x):
    y = x**3 / 12
    return (y)

def f2(x):
    y = 16 / x**4
    return(y)

plt.plot(x1, f(x1), 'b')
plt.plot(x2, f2(x2), 'r')
plt.legend()
plt.show()

tot = 0
for x in range(5000000):
    val = x / 100
    if (val <= 2): tot += f(val)
    if (val > 2): tot += f2(val)  
print("f(x) for [0,inf) is ", round(tot / 100, 2), 
      "and is a probability density")

tot = 0
for x in range(100, 500):
    val = x / 100
    if (val <= 2): tot += f(val)
    if (val > 2): tot += f2(val)    
print(tot)
print("Probability lightbulb lasts 1-5 years=",
      tot, "%")
