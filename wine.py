import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
'''The data set used is IRIS data set with the first two features'''
#from mpl_toolkits.mplot3d import Axes3D
#from numpy import vectorize
n=150
d=13
#data
#('C:/Users/sid/Desktop/MA4204/PCA/iris.txt')
iris=datasets.load_wine()
A=iris.data[:,:13]
B=iris.target

#No. of data points=n; Dimension of data=d
#Need to find vector e such that m+e*ak is optimized
m=np.mean(A,0)
S=np.zeros((d,d))
#Generating S, the scatter matrix.
for i in range (n):
    n=A[i,:]-m
    n=n[:,np.newaxis]
    S=S+np.matmul(n,np.transpose(n))

v,w=np.linalg.eig(S)
ev1=np.argmax(v)
print('Eigenvalues are the following-')
print(v)
q=v
q[ev1]=-inf
ev2=np.argmax(q)
q[ev2]=-inf
ev3=np.argmax(q)
print('Eigenvectors for the respective Eigenvalues-')
print(w)
print('The Eigenvectors we are interested in here are the ones corresponding to the 2 largest eigenvalues.')
e1=w[:,ev1]
e2=w[:,ev2]
e3=w[:,ev3]
#data reduction to 2d

Ak1=np.matmul((A-m),e1)
Ak2=np.matmul((A-m),e2)
A2=np.zeros((150,13))
for i in range(150):
    A2[i,:]=m+e1*Ak1[i]+e2*Ak2[i]
plt.scatter(Ak1,Ak2,c=B)
plt.show()
#data reduction to 3d
print('The Eigenvectors we are interested in here are the ones corresponding to the 3. largest eigenvalues.')
Ak1=np.matmul((A-m),e1)
Ak2=np.matmul((A-m),e2)
Ak3=np.matmul((A-m),e3)
A3=np.zeros((150,13))
for i in range(150):
    A3[i,:]=m+e1*Ak1[i]+e2*Ak2[i]+e3*Ak3[i]
fig=plt.figure()
ax = Axes3D(fig)
ax.scatter(Ak1,Ak2,Ak3,c=B)
plt.show()

