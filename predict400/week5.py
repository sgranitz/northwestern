## Week 5: Probability and the Media
# Birthday Problem
# https://www.washingtonpost.com/news/wonk/wp/2016/11/02/why-lifes-strangest-coincidences-really-arent-that-strange-at-all/?utm_term=.de47719120dd
# at what point would it be better than 50/50 that 2 people share a birthday. 
# They found that when there are 23 people, you have 253 potential pairs and >50% chance that one of those pairs is a match.
# To get 253 = 23! / ((23-2)! * 2!)

# Attempt to replicate in 2 methods

import numpy as np
import scipy.misc as sm
import plotly.plotly as py
import plotly.graph_objs as go

# number of guests at party
# n == 0 & n == 1 are not parties!
n = np.array([range(2,100,1)])

# Attempt 1
y  = []
y2 = []
for i in range(2,100,1):
    days = 365
    prob = 1
    for j in range(1, (i + 1), 1):
        days -=1
        prob *= days/365
    y.append(prob)
    y2.append(1 - prob)

match = go.Scatter(x = n[0], y = y2)
no_match = go.Scatter(x = n[0], y = y)
data = [match, no_match]
plot_url = py.plot(data, filename='basic-line')

better_than_50 = min(i for i in y2 if i > 0.5)
print("Party size to have >50% chance of a shared birthday =",
      y2.index(better_than_50) + 3)
# Party size to have >50% chance of a shared birthday = 23

# Attempt 2 - simplify
# Total possible number of matches
y = sm.factorial(n) / (sm.factorial(n - 2) * sm.factorial(2))

# Chance of nobody matching
a = (364/365)**y
# Chance there is a match
b = 1 - a

match    = go.Scatter(x = n[0], y = b[0])
no_match = go.Scatter(x = n[0], y = a[0])
data = [match, no_match]
plot_url = py.plot(data, filename='basic-line')

better_than_50 = min(i for i in b[0] if i > 0.5)
print("Party size to have >50% chance of a shared birthday =",
      int(np.where(b[0] == better_than_50)[0]) + 2)
#  Party size to have >50% chance of a shared birthday = 23
