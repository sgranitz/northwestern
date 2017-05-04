## Week 3: Solving Minimization Problems
# Extending the client project example from last week. 
# The project is worked by 2 types of workers (Manager, Offshore). 
# The manager worked x hours on the project 
# and offshore staff worked y hours, for at least 1200 hours. 

# Managers cost 150/hour but bring in revenue of 350/hour. 
# Offshore staff cost 35/hour but bring in revenue of 50/hour.

# We need to minimize the project cost and bring in revenue of at least 150,000. 
# Below we solve using the Simplex Method and also in Python.

# Minimize:   c = 150x + 35y
# Subject to: x + y ≥ 1,200
#             350x + 50y ≥ 150,000
# with:       x, y ≥ 0.

from scipy.optimize import linprog as lp
from matplotlib.pyplot import *
from numpy import *

c = [-150, -35]
A = [[1, 1], [350, 50]]
b = [1200, 150000]

x0_bounds = (0, None)
x1_bounds = (0, None)
res = lp(c, A_ub = A, b_ub = b,
         bounds = (x0_bounds, x1_bounds),
         options = {"disp": True})

print(res)
# fun: -76500.0
# message: 'Optimization terminated successfully.'
# nit: 2
# slack: array([ 0., 0.])
# status: 0
# success: True
# x: array([ 300., 900.])

x  = arange(0,3000,1)
y0 = arange(0,3000,1)

y1 = 1200.0 - x
y2 = 3000.0 - 7.0 * x
xlim(0,1250)
ylim(0,3500)

plot(x, y1, color = 'g')
plot(x, y2, color = 'b')
plot([300], [900], 'rs')
xfill = [0, 0, 300, 1200, 1250, 1250, 0]
yfill = [3500, 3000, 900, 0, 0, 3500, 3500]
fill_betweenx(yfill, xfill, 1250, color = 'grey', alpha = 0.2)
show()
