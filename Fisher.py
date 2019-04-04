import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB


n=100
d=4
a=0
iris=datasets.load_iris()
A=iris.data[:100,:4]
B=iris.target[:100]
m=np.mean(A,0)
A1=A[:50,:]
A2=A[50:,:]
m1=np.mean(A1,0)
m2=np.mean(A2,0)
S1=np.zeros((d,d))
S2=np.zeros((d,d))
for i in range(50):
    n1=A1[i,:]-m1
    n1=n1[:,np.newaxis]
    n2=A1[i,:]-m2
    n2=n2[:,np.newaxis]
    S1=S1+np.matmul(n1,np.transpose(n1))
    S2=S2+np.matmul(n2,np.transpose(n2))

Sw=S1+S2
w=np.matmul(np.linalg.inv(Sw),np.transpose((m1-m2)))
Data=100*np.matmul(A,w)
Du=np.ones((100,1))
plt.scatter(Data,Du,c=B)
plt.show()
plt.scatter(Data,Du,c=B)
xx=np.array([-1.1955])
yy=np.array([1])
xi=np.linspace(-4,0,100)
yi=-1.1955*np.ones(100)
plt.plot(xi,yi)
d1=Data[:50]
d2=Data[50:]
me1=np.mean(d1)
me2=np.mean(d2)
v1=np.var(d1)
v2=np.var(d2)
#-1.1955
'''Data1=np.concatenate((Data,Du),1)
p=np.linspace(-0.04, 0, 100)
clf = GaussianNB()
clf.fit(Data, B)
Pr=clf.predict_proba(p)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.scatter(Data, Du, c=B)
ax.contour(p, Du, Pr, [0.5])
xlim=(-0.04,0)
ylim=(-0.5,1.5)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.show()'''
