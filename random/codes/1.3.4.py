import numpy as np
import sys                                         
sys.path.insert(0, '/home/yogitha/random/CoordGeo')       
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
import subprocess
import shlex


#random vertices generated
A=np.array([-1,-4])
B=np.array([4,-6])
C=np.array([3,0])

D1=alt_foot(A,B,C)
E1=alt_foot(B,A,C)
F1=alt_foot(C,B,A)

H=line_intersect(norm_vec(F1,C),C,norm_vec(E1,B),B)


x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD1 = line_gen(A,D1)
x_BE1 = line_gen(B,E1)
x_CF1 = line_gen(C,F1)
x_AE1 = line_gen(A,E1)
x_AF1 = line_gen(A,F1)
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD1[0,:],x_AD1[1,:],label='$AD1$')
plt.plot(x_BE1[0,:],x_BE1[1,:],label='$BE1$')
plt.plot(x_CF1[0,:],x_CF1[1,:],label='$CF1$')
plt.plot(x_AE1[0,:],x_AE1[1,:],linestyle='dotted')
plt.plot(x_AF1[0,:],x_AF1[1,:],linestyle='dotted')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D1 = D1.reshape(-1,1)
F1 = F1.reshape(-1,1)
E1 = E1.reshape(-1,1)
H = H.reshape(-1,1)

tri_coords = np.block([[A, B, C , D1 ,E1 ,F1, H ]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C', 'D1', 'E1', 'F1', 'H']
for i, txt in enumerate(vert_labels):
    offset = 10 if txt == 'C' else -10
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, offset),
                 ha='center')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.savefig("1.3.4.png",bbox_inches='tight')
