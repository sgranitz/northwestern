## Week 8: The Fundamental Theorem of Calculus
#The Fundamental Theorem of Calculus requires that the function be continuous on a closed interval 
# before we can integrate. Find or create a function that is not continuous over some interval and 
# explain how we might still be able to integrate the function. Using Python, incorporate a graph 
# of your function that also indicates the area under the curve. Be sure to share your Python code and output.

from fillplots import plot_regions, annotate_regions
import numpy as np

a = 0
b = 10

n = 10
upper = n - (n % 10)
delta = (b - a) / n
area  = 0

def f(x):
    return int(x)

for i in range(0, upper): 
    area += f(i * delta) * delta
print("Area from", a, "to", b, "is", np.ceil(area))

# Area from 0 to 10 is 45.0

plotter = plot_regions([
    [(lambda x: np.floor(x), True)], 
], xlim=(a, b), ylim=(a, b))
annotate_regions(plotter.regions, 'Area')
