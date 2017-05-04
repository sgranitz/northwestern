## Week 4: Game Applications
# Battleship (http://www.datagenetics.com/blog/december32011/)
# probability of a perfect game. This would be sinking your 
# opponent's ships without missing a single shot. 
# Since there are 17 targets on the 100 spaces, the probability of this would be:
# 17/100 * 16/99 * 15/98 * ... * 1/84 = 355,687,428,096,000 / 2,365,369,369,446,553,061,560,941,772,800,000

import math
def test(x, y):
    a = math.factorial(x)
    b = 1
    while x > 0:
        x -= 1
        b *= (y - x) 
    return(a/b)
print(test(17, 100))
# 1.5037289004009696e-19

print(355687428096000 / 2365369369446553061560941772800000)
# 1.5037289004009696e-19

# After this the article simulates 100MM games of battleship using a random selection strategy. 
# Half of the games took 96 or more shots. 

def game(num):
    a = []
    for i in range(1, (num + 1)):
        empty = [0] * (100 - 17)
        filled = [1] * 17
        p0 = empty + filled
        p1 = random.sample(p0, len(p0))
        a.extend([100 - p1[::-1].index(1)])
    return(a)

# This will randomly distribute the 17 hits (or filled spaces on the board) and tell me how many hits 
# it took to get the final shot and win the game. I can then run loop for as many games as I want to simulate. 
# First we can test it with 10 games:

print(game(10))
# [88, 97, 98, 98, 94, 96, 93, 99, 84, 93]

# Games are taking nearly 100 shots to win. Let's try with 500K games:
import numpy as np
import matplotlib.pyplot as plt

res = game(500000)
print(min(res))
# 50 - the fastest I was able to win was 50 shots. Not bad but the author found their fastest at 44.

res2 = [i for i in res if i > 95]
per_over_95 = len(res2) / len(res)
print(per_over_95)
# 0.614768 - The author had ~50% of their games at 96+ shots whereas I am finding more than 60%.

res.sort()
print(res[(int(-0.99 * 500000))])
# 79 - 99% of wins took 79 shots compared to author's 78. 
a = np.hstack(res)
plt.hist(a, bins=25)
plt.show()
