# 1. Reebok is designing a new type of Crossfit shoe, the Nano X. The fixed cost for the
# production will be $24,000. The variable cost will be $36 per pair of shoes. The shoes will
# sell for $107 for each pair. Using Python, graph the cost and revenue functions and
# determine how many pairs of sneakers will have to be sold for the company to break even on
# this new line of shoes.

import matplotlib.pyplot as plt
import math

# Expenses
# y = 36x + 24000
exp_slope = 36
exp_int   = 24000

exp_x0 = 0
exp_y0 = exp_slope * exp_x0 + exp_int
exp_x1 = 1000
exp_y1 = exp_slope * exp_x1 + exp_int

# Revenue
# y = 107x
rev_slope = 107
rev_int   = 0

rev_x0 = 0
rev_y0 = rev_slope * rev_x0 + rev_int
rev_x1 = 1000
rev_y1 = rev_slope * rev_x1 + rev_int

# Breakeven
# 107x = 36x + 24000
#  71x = 24000
#    x = 338.028
#    y = 107(338.028) = 36169.014

be_x = 24000 / 71
be_y = 107 * be_x

# Plot the lines
fig, shoe = plt.subplots()
shoe.scatter([exp_x0, exp_x1], 
             [exp_y0, exp_y1], 
             c = 'r')
shoe.plot([exp_x0, exp_x1], 
          [exp_y0, exp_y1], 
          c = 'r', alpha = 0.3)

shoe.scatter([rev_x0, rev_x1], 
             [rev_y0, rev_y1], 
             c = 'g')
shoe.plot([rev_x0, rev_x1], 
          [rev_y0, rev_y1], 
          c = 'g', alpha = 0.3)

shoe.scatter([be_x],
             [be_y], 
             c = 'b', s = 100)
plt.xlim(0, 750)
plt.ylim(0, 75000)
plt.show()

print("To break even, Reebok must sell", 
      math.ceil(be_x), "shoes.")

# 2. Nicole invests a total of $17,500 in three products. She invests one part in a mutual fund
# which has an annual return of 11%. She invests the second part in government bonds at 7%
# per year. The third part she puts in CDs at 5% per year. She invests twice as much in the
# mutual fund as in the CDs. In the first year Nicole's investments bring a total return of $1495.
# How much did she invest in each product?

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
# x + y + z = 17500
# 0.11x + 0.07y + 0.05z = 1495
# x - 2z = 0

a = inv(np.matrix('1 1 1; 11 7 5; 1 0 -2'))
b = np.array([17500, 149500, 0])
res = a.dot(b)
print("mutual funds=", res[0, 0],
      "gov't bonds =", res[0, 1],
      "CDs         =", res[0, 2])

labels = 'Mut Funds', "Gov't Bonds", 'CDs'
sizes  =  [res[0, 0], res[0, 1], res[0, 2]]
colors = ['lightskyblue', 'pink', 'yellowgreen']

plt.pie(sizes, labels = labels, colors = colors,
        autopct = '%1.1f%%', startangle = 140)
 
plt.axis('equal')
plt.show()

# 3. A company has 252 sales reps, each to be assigned to one of four marketing teams. If the first
# team is to have three times as many members as the second team and the third team is to
# have twice as many members as the fourth team, how can the members be distributed among
# the teams?

import pandas as pd
# w + x + y + z = 252
# w = 3x
# y = 2z
# 3x + x + 2z + z = 252
# 4x + 3z = 252
# 4x = 252 - 3z
#  x = 63 - 3/4z
#  w = 3 * (63 - 3/4z)
#  w = 189 - 9/4z

res = []
for z in range(253):
    z = float(z)
    x = float(63 - 3 * z / 4)
    y = float(2 * z)
    w = float(189 - 9 * z / 4)
    a,b = False,False
    if (w > 0) & (x > 0) & (y > 0) & (z > 0):
        a = True
    if (w.is_integer()) & (x.is_integer()) & (y.is_integer()) & (z.is_integer()):
        b = True
    if a & b: res.append([w, x, y, z])

teams = ['team1', 'team2', 'team3', 'team4']
print(pd.DataFrame(res, columns = teams))
pd.DataFrame(res, columns = teams).plot(kind = 'bar', stacked = True)

# 4. A company makes three types of artisanal chocolate bars: cherry, almond, and raisin. Matrix
# A gives the amount of ingredients in one batch. Matrix B gives the costs of ingredients from
# suppliers J and K. Using Python, calculate the cost of 100 batches of each candy using
# ingredients from supplier K.

import numpy as np
a = np.matrix('6 8 1; 6 4 1; 5 7 1')
b = np.matrix('4 3; 4 5; 2 2')

batch = a.dot(b)
print("100 cherry =", batch[0, 1] * 100,
      "100 almond =", batch[1, 1] * 100,
      "100 raisin =", batch[2, 1] * 100)

# 5. Welsh-Ryan Arena seats 15,000 people. Courtside seats cost $8, first level seats cost $6, and
# upper deck seats cost $4. The total revenue for a sellout is $76,000. If half the courtside seats,
# half the upper deck seats, and all the first level seats are sold, then the total revenue is
# $44,000. How many of each type of seat are there?

import numpy as np
from numpy.linalg import inv
#  x +  y +  z = 15000
# 8x + 6y + 4z = 76000
# 0.5(8x + 4z) + 6y = 44000
# 4x + 6y + 2z = 44000

a = inv(np.matrix('1 1 1; 8 6 4; 4 6 2'))
b = np.array([15000, 76000, 44000])
res = a.dot(b)
print("courtside  =", res[0, 0],
      "first level=", res[0, 1],
      "upper deck =", res[0, 2])

# 6. Due to new environmental restrictions, a chemical company must use a new process to
# reduce pollution. The old process emits 6 g of Sulphur and 3 g of lead per liter of chemical
# made. The new process emits 2 g of Sulphur and 4 g of lead per liter of chemical made. The
# company makes a profit of 25¢ per liter under the old process and 16¢ per liter under the new
# process. No more than 18,000 g of Sulphur and no more than 12,000 g of lead can be emitted
# daily. How many liters of chemicals should be made daily under each process to maximize
# profits? What is the maximum profit?

from scipy.optimize import linprog as lp
import numpy as np
# maximize: 0.25x + 0.16y
# subject to:
#   6x + 2y <= 18000
#   3x + 4y <= 12000
#      x, y >= 0

A = np.array([[6,  2], [3, 4]])
b = np.array([18000, 12000])
liters = lp(np.array([-0.25, -0.16]), A, b)

print("old method=", round(liters.x[0], 2), "liters.",
      "new method=", round(liters.x[1], 2), "liters.")
print("Max daily profit=", 
      round(0.25 * liters.x[0] + 0.16 * liters.x[1], 2))

# 7. Northwestern is looking to hire teachers and TA’s to fill its staffing needs for its summer
# program at minimum cost. The average monthly salary of a teacher is $2400 and the average
# monthly salary of a TA is $1100. The program can accommodate up to 45 staff members and
# needs at least 30 to run properly. They must have at least 10 TA’s and may have up to 3 TA’s
# for every 2 teachers. Using Python, find how many teachers and TA’s the program should
# hire to minimize costs. What is the minimum cost?

from scipy.optimize import linprog as lp
import numpy as np
# minimize: 2400x + 1100y
# subject to:
#   x + y <= 45
#   x + y >= 30
#       y >= 10
#      2y <= 3x
#    x, y >= 0

A = np.array([[-1, -1], [-3, 2]])
b = np.array([-30, 0])
x_bounds = (0,  45)
y_bounds = (10, 45)
hire = lp(np.array([2400, 1100]), A, b,
          bounds = (x_bounds, y_bounds))

print("Hire", hire.x[0], "teachers.",
      "Hire", hire.x[1], "TAs.")
print("Minimum cost=", 
      2400 * hire.x[0] + 1100 * hire.x[1])

# 8. To be at his best as a teacher, Roger needs at least 10 units of vitamin A, 12 units of vitamin
# B, and 20 units of vitamin C per day. Pill #1 contains 4 units of A and 3 of B. Pill #2 contains
# 1 unit of A, 2 of B, and 4 of C. Pill #3 contains 10 units of A, 1 of B, and 5 of C. Pill #1 costs
# 6 cents, pill #2 costs 8 cents, and pill #3 costs 1 cent. How many of each pill must Roger take
# to minimize his cost, and what is that cost?

from scipy.optimize import linprog as lp
import numpy as np
# minimize: 0.06x + 0.08y + 0.01z
# subject to:
#   4x +  y + 10z >= 10
#   3x + 2y +   z >= 12
#        4y +  5z >= 20
#         x, y, z >= 0

A = np.array([[-4, -1, -10], [-3, -2, -1], [0, -4, -5]])
b = np.array([-10, -12, -20])
pills = lp(np.array([0.06, 0.08, 0.01]), A, b)

print("Pill #1=", pills.x[0],
      "Pill #2=", pills.x[1],
      "Pill #3=", pills.x[2],)
print("Minimum cost=", 
      0.06 * pills.x[0] + 0.08 * pills.x[1] + 0.01 * pills.x[2])

# 9. An electronics store stocks high-end DVD players, surround sound systems, and televisions.
# They have limited storage space and can stock a maximum of 210 of these three machines.
# They know from past experience that they should stock twice as many DVD players as stereo
# systems and at least 30 television sets. If each DVD player sells for $450, each surround
# sound system sells for $2000, and each television sells for $750, how many of each should be
# stocked and sold for maximum revenues? What is the maximum revenue?

from scipy.optimize import linprog as lp
import numpy as np
# maximize: 450x + 2000y + 750z
# subject to:
#   x + y + z <= 210
#   x >= 2y
#   z >= 30
#   x, y, z >= 0

A = np.array([[1, 1, 1], [-1, 2, 0], [0, 0, -1]])
b = np.array([210, 0, -30])
units = lp(np.array([-450, -2000, -750]), A, b)

print("DVDs =", units.x[0],
      "SS Systems=", units.x[1],
      "TVs =", units.x[2],)
print("Maximum revenue=", 
      450 * units.x[0] + 2000 * units.x[1] + 750 * units.x[2])

# 10. A fast-food company is conducting a sweepstakes, and ships two boxes of game pieces to a
# particular franchise. Box A has 4% of its contents being winners, while 5% of the contents of
# box B are winners. Box A contains 27% of the total tickets. The contents of both boxes are
# mixed in a drawer and a ticket is chosen at random. Using Python, find the probability it
# came from box A if it is a winner.

is_box_a  = 0.27
box_a_win = 0.04
box_b_win = 0.05

a = is_box_a * box_a_win
b = (1 - is_box_a) * box_b_win
prob = a / (a + b)

print("Probability that winner came from Box A is",
      round(prob * 100, 3), "%.")

# Heatmap of probabilities for drawing random card
import pandas as pd
import seaborn as sns

a2 = is_box_a * (1 - box_a_win)
b2 = (1 - is_box_a) * (1 - box_b_win)
prob2 = a  / (a + b + a2 + b2)
prob3 = b  / (a + b + a2 + b2)
prob4 = a2 / (a + b + a2 + b2)
prob5 = b2 / (a + b + a2 + b2)

df = pd.DataFrame([[prob2, prob4], [prob3, prob5]], 
                  index = ['Box A', 'Box B'], 
                  columns = ['Winner', 'Loser'])
sns.heatmap(df)
