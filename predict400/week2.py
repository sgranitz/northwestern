## Week 2: Linear Equations and the Echelon or Gauss-Jordan Methods
# I used an example of a client project. The numbers are fake to simplify for the discussion. 
# The project is worked by 3 types of workers (Manager, Staff, Offshore).

# The manager worked x hours on the project, onshore staff worked y hours, 
# and offshore staff worked z hours, for a total of 1300 hours.
# Managers cost 150/hour but bring in revenue of 350/hour. 
# Onshore staff cost 80/hour but bring in revenue of 160/hour. 
# Finally, offshore staff cost 35/hour but bring in revenue of 50/hour.

# We know the project cost 96,750 and brought in revenue of 195,000.

# Echelon:
import numpy as np

a = np.array([1,1,1,1300])
b = np.array([150,80,35,96750])
c = np.array([350,160,50,195000])

b = (b - a * 150) * -1

c = (c - a * 350) * -1
c = (c - b * 19.0 / 7) * -1
c = c * 7.0/85

b = b / 70.0

z = round(c[3], 0)
y = round(b[3] - b[2] * z, 0)
x = a[3] - y - z

print (x,y,z)
# (250.0, 500.0, 550.0)

# Gauss-Jordan:
from numpy.linalg import inv

a = inv(np.matrix('1 1 1; 150 80 35; 350 160 50'))
b = np.array([1300, 96750, 195000])
a.dot(b)
# matrix([[ 250.,  500.,  550.]])
