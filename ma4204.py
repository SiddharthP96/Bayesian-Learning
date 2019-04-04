import numpy as np
import matplotlib.pyplot as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from numpy import vectorize 


#Case 1- Variance Matrix=IxVariance  (All features have the same variance)
x1=np.array([[2,2]])
y1=np.array([[2,0],[0,2]])
p1=0.4
x2=np.array([[0,0]])
y2=np.array([[2,0],[0,2]])
p2=0.6
#Conditional Probability for class 1
def nor1(a,b):
    x=np.array([[a,b]])
    p=0
    d=np.linalg.det(y1)
    p1=1/(2*pi*sqrt(d))
    p=p1*exp((-0.5)*np.matmul(np.matmul((x-x1),np.linalg.inv(y1)),np.transpose(x-x1)))
    return p
#Conditional Probability for class 2
norm1=np.vectorize(nor1)
def nor2(a,b):
    x=np.array([[a,b]])
    p=0
    d=np.linalg.det(y2)
    p1=1/(2*pi*sqrt(d))
    p=p1*exp((-0.5)*np.matmul(np.matmul((x-x2),np.linalg.inv(y2)),np.transpose(x-x2)))
    return p
norm2=np.vectorize(nor2)
#discriminant function
def g(a,b):
    e=log((nor1(a,b)*p1))-log((nor2(a,b)*p2))
    return e

#if g>0 choose class 1 else class 2
gv = np.vectorize(g)

p = np.linspace(-5, 5, 40)
q = np.linspace(-5, 5, 40)

X,Y=np.meshgrid(p,q)

fig = plt.figure()
ax = Axes3D(fig)
Z = norm2(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
Z = norm1(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)

# Plot the surface.

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 0.2)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'p(x,y)')

plt.show()
fig = plt.figure()
ax = Axes3D(fig)
Z = gv(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
Z1 = np.zeros((40,40))
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=plt.cm.Greys_r)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-10, 10)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'g(x,y)')
plt.show()

plt.figure()
CS = plt.contour(X, Y, Z, 0)              
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Case 1')

plt.show()



#Case 2- Equal Variance
x1=np.array([[2,2]])
y1=np.array([[2,0.6],[0.6,1]])

x2=np.array([[0,0]])
y2=np.array([[2,0.6],[0.6,1]])


#if g>0 choose class 1 else class 2

p = np.linspace(-5, 5, 40)
q = np.linspace(-5, 5, 40)

X,Y=np.meshgrid(p,q)

fig = plt.figure()
ax = Axes3D(fig)
Z = norm2(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
Z = norm1(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
# Plot the surface.
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 0.2)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'p(x,y)')

plt.show()
fig = plt.figure()
ax = Axes3D(fig)
Z = gv(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
Z1 = np.zeros((40,40))
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=plt.cm.Greys_r)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-10, 10)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'g(x,y)')
plt.show()

plt.figure()
CS = plt.contour(X, Y, Z, 0)              
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Case 2')

plt.show()
#Case 3- Generalised

x1=np.array([[2,2]])
y1=np.array([[2,0.6],[0.6,1]])

x2=np.array([[0,0]])
y2=np.array([[1.5,0.7],[0.7,2]])
#Conditional Probability for class 1

#if g>0 choose class 1 else class 2

p = np.linspace(-5, 5, 40)
q = np.linspace(-5, 5, 40)

X,Y=np.meshgrid(p,q)

fig = plt.figure()
ax = Axes3D(fig)
Z = norm2(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
Z = norm1(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
# Plot the surface.
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 0.2)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'p(x,y)')

plt.show()
fig = plt.figure()
ax = Axes3D(fig)
Z = gv(X,Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
Z1 = np.zeros((40,40))
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=plt.cm.Greys_r)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-10, 10)
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_zlabel(r'g(x,y)')
plt.show()

plt.figure()
CS = plt.contour(X, Y, Z, 0)              
plt.clabel(CS, fontsize=9, inline=1)
plt.title('Case 3')
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
plt.show()
