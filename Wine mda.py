import numpy as np
import matplotlib.pyplot as plt
from math import *
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB
n=178
d=13
a=0
iris=datasets.load_wine()
A=iris.data[:,:]
B=iris.target
m=np.mean(A,0)
A1=A[:59,:]
A2=A[59:130,:]
A3=A[130:,:]
m1=np.mean(A1,0)
m2=np.mean(A2,0)
m3=np.mean(A3,0)
#59,71,48
S1=np.zeros((d,d))
S2=np.zeros((d,d))
S3=np.zeros((d,d))

n1=A1-m1
n2=A2-m2
n3=A3-m3

S1=S1+np.matmul(np.transpose(n1),n1)
S2=S2+np.matmul(np.transpose(n2),n2)
S3=S3+np.matmul(np.transpose(n3),n3)
Sw=S1+S2+S3
Sb1=np.zeros((d,d))
Sb2=np.zeros((d,d))
Sb3=np.zeros((d,d))

n1=m1-m
n2=m2-m
n3=m3-m
n2=n2[np.newaxis,:]
n3=n3[np.newaxis,:]
n1=n1[np.newaxis,:]
Sb1=Sb1+np.matmul(np.transpose(n1),n1)
Sb2=Sb2+np.matmul(np.transpose(n2),n2)
Sb3=Sb3+np.matmul(np.transpose(n3),n3)
Sb=59*Sb1+71*Sb2+48*Sb3

M=np.linalg.inv(Sw)
Mat=np.matmul(M,Sb)
v,w=np.linalg.eig(Mat)
'''print(v)
print(w)'''
W=w[:,1:3]
NewData=np.matmul(A,W)
x=NewData[:,0]
y=NewData[:,1]
plt.scatter(x,y,c=B)
plt.show()
p=np.linspace(-6, 3, 100)
q=np.linspace(-6, 3, 100)
S1,S2=np.meshgrid(p,q)

clf = GaussianNB()
clf.fit(NewData, B)

    
Pr=clf.predict_proba(np.c_[S1.ravel(), S2.ravel()])
Pr1 = Pr[:, 0].reshape(S1.shape)
Pr2 = Pr[:, 1].reshape(S1.shape)
Pr3 = Pr[:, 2].reshape(S1.shape)
#Plot
fig = plt.figure(figsize=(5, 3.75))
ax = fig.add_subplot(111)
ax.scatter(NewData[:,0], NewData[:, 1], c=B)
ax.contour(S1, S2, Pr1, [0.5])
ax.contour(S1, S2, Pr2, [0.5])
ax.contour(S1, S2, Pr3, [0.5])
xlim=(-6,3)
ylim=(-6,3)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.show()


