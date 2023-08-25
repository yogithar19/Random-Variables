import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.insert(0, '/home/yogitha/random/CoordGeo')
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

omat = np.array([[0, 1], [-1, 0]])

#random vertices generated
A=np.array([-1,-4])
B=np.array([4,-6])
C=np.array([3,0])

t1 = norm_vec(B,C) 
n1 = t1/np.linalg.norm(t1) 
t2 = norm_vec(C,A)
n2 = t2/np.linalg.norm(t2)
t3 = norm_vec(A,B)
n3 = t3/np.linalg.norm(t3)

#parameters
n_a=dir_vec(n3,n2)
n_b=dir_vec(n1,n3)
n_c=dir_vec(n1,n2)

m_a=norm_vec(n2,n3)
m_b=norm_vec(n1,n3)
m_c=norm_vec(n1,n2)
c_a=n_a@A
c_b=n_b@B
c_c=n_c@C

I,r=icircle(A,B,C)


#finding k for D_3,E_3 and F_3
k2=((I-A)@(A-B))/((A-B)@(A-B))
k1=((I-A)@(A-C))/((A-C)@(A-C))
k3=((I-B)@(B-C))/((B-C)@(B-C))
#finding D_3,E_3 and F_3
E3=A+(k1*(A-C))
F3=A+(k2*(A-B))
D3=B+(k3*(B-C))
incircle=circ_gen(I,r)
#plot
plt.plot(incircle[0,:],incircle[1,:],label='$incircle$')
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AI = line_gen(A,I)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D3 = D3.reshape(-1,1)
E3 = E3.reshape(-1,1)
F3 = F3.reshape(-1,1)
I = I.reshape(-1,1)
tri_coords = np.block([[A,B,C,D3,E3,F3,I]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','$D_3$','$E_3$','$F_3$','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.savefig("1.5.png",bbox_inches='tight')


