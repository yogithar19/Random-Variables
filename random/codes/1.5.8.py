import numpy as np
import sys                                         
sys.path.insert(0, '/home/yogitha/random/CoordGeo')       
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

A = np.array([1, -1])
B = np.array([-4, 6])
C = np.array([-3, -5])

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Generating the incircle
[I,r] = icircle(A,B,C)
x_icirc= circ_gen(I,r)

k=((I-B)@(C-B))/((C-B)@(C-B))
D3=B+k*(C-B)
k1=((I-C)@(A-C))/((A-C)@(A-C))
E3=C+k1*(A-C)
k2=((I-A)@(B-A))/((B-A)@(B-A))
F3=A+k2*(B-A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
# plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')

#Plotting the incircle
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
plt.plot(D3[0],D3[1],label='$D_3$')
plt.plot(E3[0],E3[1],label='$E_3$')
plt.plot(F3[0],F3[1],label='$F_3$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
I = I.reshape(-1,1)
D3=D3.reshape(-1,1)
E3=E3.reshape(-1,1)
F3=F3.reshape(-1,1)

#Labeling the coordinates
tri_coords = np.block([[A,B,C,D3,I, E3,F3]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D3','I','E3', 'F3']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.savefig("1.5.8.png",bbox_inches='tight')

