import numpy as np
import matplotlib.pyplot as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from numpy import vectorize
import matplotlib.patches as mpatches
x1=-0.6498
y1=0.01170
p1=0.5
x2=-2.6381
#x is the mean and y is variance
y2=0.0887
p2=0.5
#Conditional Probability for class 1
def nor1(a):
    p=(1/sqrt(2*y1*pi))*exp(-((a-x1)**2)/(2*y1))
    return p
#Conditional Probability for class 2
norm1=np.vectorize(nor1)
def nor2(a):
    p=(1/sqrt(2*y2*pi))*exp(-((a-x2)**2)/(2*y2))
    return p
norm2=np.vectorize(nor2)
#discriminant function
def g(a):
    e=log((nor1(a)*p1))-log((nor2(a)*p2))
    return e
gv=np.vectorize(g)
a=np.linspace(-4,0,100)

plt.plot(a,norm2(a), 'r', a, norm1(a), 'blue', a, gv(a)/100, 'orange')
plt.xlabel('x')
plt.ylabel('p(x)')

o_patch = mpatches.Patch(color='orange', label='g(x)')
plt.legend(handles=[o_patch])
plt.axhline(0, color='black')
plt.axvline(0, color='black')
'''ylim=(-0.5,1.5)
plt.set_ylim(ylim)'''
plt.show()

