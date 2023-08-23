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

np.set_printoptions(precision=2)
A= np.array([1,-1])
B= np.array([-4,6])
C= np.array([-3,-5])

D=(B+C)/2
E=(C+A)/2
F=(A+B)/2

G=line_intersect(norm_vec(F,C),C,norm_vec(E,B),B)

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
#x_AD = line_gen(A,D)
x_BE = line_gen(B,E)
x_CF = line_gen(C,F)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
#plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
plt.plot(x_CF[0,:],x_CF[1,:],label='$CF$')

A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
F = F.reshape(-1,1)
E = E.reshape(-1,1)
G = G.reshape(-1,1)
tri_coords = np.block([[A, B, C , D ,E ,F , G]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
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

plt.savefig("1.2.3.png",bbox_inches='tight')
