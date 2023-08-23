import sys
sys.path.insert(0, '/home/yogitha/random/CoordGeo')  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen

A = np.array([1, -4])
B = np.array([4, -6])
C = np.array([3, 0])

omat = np.array([[0,1],[-1,0]])

def dir_vec(A,B):
   return B-A
  
def length(A,B):
   return np.sqrt(A@B)
   
def norm(X,Y):
    magnitude=round(float(np.linalg.norm([X-Y])),3)
  
def normal(x,y):
   return x@y
  
def angle(m1,m2):
    dot=m1.T@m2
    norm=np.linalg.norm(m1)*np.linalg.norm(m2)
    return np.degrees(np.arccos(dot/norm))
    
#intersection of lines
def line_intersect(n1,A1,n2,A2):
	N=np.block([[n1],[n2]])
	p = np.zeros(2)
	p[0] = n1@A1
	p[1] = n2@A2
	#Intersection
	P=np.linalg.inv(N)@p
	return P  
	
def alt_foot(A,B,C):
  m = B-C
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P
	
   
# 1 direction vectors  
print("1.1.1")
v_AB=dir_vec(A,B)
print("Direction vector of AB:",v_AB)
v_BC=dir_vec(B,C)
print("Direction vector of BC:",v_BC)
v_CA=dir_vec(C,A)
print("Direction vector of CA:",v_CA)

#transpose
t_AB=np.array(v_AB).T
t_BC=np.array(v_BC).T
t_CA=np.array(v_CA).T

# 2 length of sides
AB=length(t_AB , v_AB)
print("\n1.1.2")
print("Length of AB=",AB)
BC=length(t_BC,v_BC)
print("Length of BC=", BC)
CA=length(t_CA, v_CA)
print("Length of CA=",CA)

#3
print("\n1.1.3")
Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])
rank = np.linalg.matrix_rank(Mat)
if (rank<=2):
	print("Hence proved that points A,B,C in a triangle are collinear")
else:
	print("The given points are not collinear")
	
#x_AB = line_gen(A,B)
#x_BC = line_gen(B,C)
#x_CA = line_gen(C,A)
#plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
#plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
#plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
#A = A.reshape(-1,1)
#B = B.reshape(-1,1)
#C = C.reshape(-1,1)
#tri_coords = np.block([[A, B, C]])
#plt.scatter(tri_coords[0, :], tri_coords[1, :])
#vert_labels = ['A', 'B', 'C']
#for i, txt in enumerate(vert_labels):
#    offset = 10 if txt == 'C' else -10
#    plt.annotate(txt,
#                 (tri_coords[0, i], tri_coords[1, i]),
#                 textcoords="offset points",
#                 xytext=(0, offset),
#                 ha='center')
#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.legend(loc='best')
#plt.grid() # minor
#plt.axis('equal')
#plt.savefig("1.1.3.png",bbox_inches='tight')

	
	
#4 parametric equations:
print("\n1.1.4")
print("parametric of AB form is :""x=",A,"+ k",v_AB)
print("parametric of BC form is :""x=",B,"+ k",v_BC)
print("parametric of CA form is :""x=",C,"+ k",v_CA)

#5 
n_AB=normal(omat,v_AB)
n_BC=normal(omat,v_BC)
n_CA=normal(omat,v_CA)

print("\n1.1.5")
print("Normal form of AB",n_AB,"x=",np.array(n_AB).T@A)
print("Normal form of BC",n_BC,"x=",np.array(n_BC).T@B)
print("Normal form of CA",n_CA,"x=",np.array(n_CA).T@C)

#6
#cross_product calculation
cross_product = np.cross(v_AB,v_CA)
#magnitude calculation
magnitude = np.linalg.norm(cross_product)
print("\n1.1.6")
print("Area=",0.5*magnitude)

#1.1.7
dotA=(C-A).T@(B-A)
NormA=(np.linalg.norm(B-A))*(np.linalg.norm(C-A))
print("\n1.1.7")
print('value of angle A: ', np.degrees(np.arccos((dotA)/NormA)))


dotB=(A-B).T@(C-B)
NormB=(np.linalg.norm(A-B))*(np.linalg.norm(C-B))
print('value of angle B: ', np.degrees(np.arccos((dotB)/NormB)))

dotC=(A-C).T@(B-C)
NormC=(np.linalg.norm(A-C))*(np.linalg.norm(B-C))
print('value of angle C: ', np.degrees(np.arccos((dotC)/NormC)))

#1.2.1
D = (B + C)/2
E = (A + C)/2
F = (A + B)/2

print("\n 1.2.1")
print("D:", D)
print("E:", E)
print("F:", F)

#1.2.2
n_AD=omat@(D-A)
n_BE=omat@(E-B)
n_CF=omat@(F-C)

print("\n1.2.2")
print("Equation of AD:", n_AD,"x=",n_AD@A)
print("Equation of BE:", n_BE,"x=",n_BE@B)
print("Equation of CF:", n_CF,"x=",n_CF@C)

#1.2.3
G=line_intersect(n_CF,C,n_BE,B)
print("\n1.2.3")
print("G=",G)

#1.2.4
AG = np.linalg.norm(G - A)
GD = np.linalg.norm(D - G)

BG = np.linalg.norm(G - B)
GE = np.linalg.norm(E - G)
 
CG = np.linalg.norm(G - C)
GF = np.linalg.norm(F - G)

print("1.2.4")
print("AG/GD= ",round((AG/GD),0))
print("BG/GE= ",round((BG/GE),0))
print("CG/GF= ",round((CG/GF),0))

#1.2.5
Mat = np.array([[1,1,1],[A[0],G[0],D[0]],[A[1],G[1],D[1]]])
rank = np.linalg.matrix_rank(Mat)
if (rank<=2):
	print("1.2.5\nA,G,D are collinear")
else:
	print("1.2.5\nThe given points are not collinear")

print("\n1.2.6")
#1.2.6
G0=(A+B+C)/3
if (int(G0.all()) == int(G.all())): 
   print("hence verified G=(A+B+C)/3")
else:
   print("not verified")
print("\n1.2.7")

#1.2.7
LHS=(A-F)
RHS=(E-D)
#checking LHS and RHS 
if LHS.all()==RHS.all() :
   print("A-F=E-D and AFED is a parallelogram")
else:
    print("Not equal")
print("\n1.3.1")	
#1.3.1

D1=alt_foot(A,B,C)
print(D1)
E1=alt_foot(B,C,A)
print(E1)
F1=alt_foot(C,A,B)
print(F1)
print("Normal vector of AD1=", v_BC)

print("\n1.3.2")
#1.3.2
print("Equation of AD1:", t_BC, "x=", t_BC@A)

print("\n1.3.3")
#1.3.3
print("Equation of BE1:", t_CA, "x=", t_CA@B)
print("Equation of CF1:", t_AB, "x=", t_AB@C)

print("\n1.3.4")
#1.3.4
H=line_intersect(v_CA,B,v_AB,C)
print("H=",H)

print("\n1.3.5")
#1.3.5
result = int(((A - H).T) @ (B - C))    # Checking orthogonality condition...

# printing output
if result == 0:
  print("(A - H)^T (B - C) = 0\nHence Verified.")

else:
  print("(A - H)^T (B - C)) != 0\nHence the given statement is wrong.")
 
print("\n1.4.1")
#1.4.1
print("Perpendicular bisector of AB:",t_AB,"x=",t_AB@F)
print("Perpendicular bisector of BC:",t_BC,"x=",t_BC@D)
print("Perpendicular bisector of CA:",t_CA,"x=",t_CA@E)

#1.4.2
print("\n1.4.2")
O=line_intersect(t_AB,F,t_CA,E)
print("O=",O)

#1.4.3
print("\n1.4.3")
OD=(O-D).T
if(int(OD@v_BC)==0):
    print("satisfies")
else:
    print("doesnt")
    
#1.4.4
print("\n1.4.4")
OA=np.linalg.norm(A-O)
OB=np.linalg.norm(B-O)
OC=np.linalg.norm(C-O)

if(OA.all()==OB.all()==OC.all()):
    print("OA=OB=OC=", OA)

    
#1.4.5 draw circle
r=OA

#1.4.6
print("\n1.4.6")
#To find angle BOC
dot_pt_O = (B - O) @ ((C - O).T)
norm_pt_O = np.linalg.norm(B - O) * np.linalg.norm(C - O)
cos_theta_O = dot_pt_O / norm_pt_O
angle_BOC =360 - round(np.degrees(np.arccos(cos_theta_O)),4)  #Round is used to round of number till 5 decimal places
print("angle BOC = " + str(angle_BOC))

#To find angle BAC
dot_pt_A = (B - A) @ ((C - A).T)
norm_pt_A = np.linalg.norm(B - A) * np.linalg.norm(C - A)
cos_theta_A = dot_pt_A / norm_pt_A
angle_BAC = round(np.degrees(np.arccos(cos_theta_A)),4)  #Round is used to round of number till 5 decimal places
print("angle BAC = " + str(angle_BAC))
#To check whether the answer is correct
if angle_BOC == 2 * angle_BAC:
  print("\nangle BOC = 2 times angle BAC\nHence the give statement is correct")
else:
  print("\nangle BOC ≠ 2 times angle BAC\nHence the given statement is wrong")



#To find angle AOC
dot_pt_O = (A - O) @ ((C - O).T)
norm_pt_O = np.linalg.norm(A - O) * np.linalg.norm(C - O)
cos_theta_O = dot_pt_O / norm_pt_O
angle_AOC =round(np.degrees(np.arccos(cos_theta_O)),4)  #Round is used to round of number till 5 decimal places
print("angle AOC = " + str(angle_AOC))

#To find angle ABC
dot_pt_A = (A - B) @ ((C - B).T)
norm_pt_A = np.linalg.norm(A - B) * np.linalg.norm(C - B)
cos_theta_A = dot_pt_A / norm_pt_A
angle_ABC = round(np.degrees(np.arccos(cos_theta_A)),4)  #Round is used to round of number till 5 decimal places
print("angle ABC = " + str(angle_ABC))
  
  #To find angle BOC   AOB
dot_pt_O = (A - O) @ ((B - O).T)
norm_pt_O = np.linalg.norm(A - O) * np.linalg.norm(B - O)
cos_theta_O = dot_pt_O / norm_pt_O
angle_AOB =round(np.degrees(np.arccos(cos_theta_O)),4)  #Round is used to round of number till 5 decimal places
print("angle A0B = " + str(angle_AOB))

#To find angle BAC BCA
dot_pt_A = (B - C) @ ((A - C).T)
norm_pt_A = np.linalg.norm(B - C) * np.linalg.norm(A - C)
cos_theta_A = dot_pt_A / norm_pt_A
angle_BCA = round(np.degrees(np.arccos(cos_theta_A)),4)  #Round is used to round of number till 5 decimal places
print("angle BCA = " + str(angle_BCA))

#1.4.7
print("\n1.4.7")

#1.5.1
print("\n1.5.1")
print("angle bisector of A:", (n_AB/AB-n_CA/CA).T,"x=", (((n_AB).T@A)/AB)-(((n_CA).T@A)/CA))
print("angle bisector of B:", (n_BC/BC-n_AB/AB).T,"x=", (((n_BC).T@B)/BC)-(((n_AB).T@B)/AB))
print("angle bisector of C:", (n_CA/CA-n_BC/BC).T,"x=", (((n_CA).T@C)/CA)-(((n_BC).T@C)/BC))

#1.5.2
print("\n1.5.2")
s1=n_BC/BC-n_AB/AB
s2=n_CA/CA-n_BC/BC
I=line_intersect(s1,B,s2,C)
print("incentre=",I)

#1.5.3
print("\n1.5.3")
#BAI
bai=angle((B-A),(I-A))
print("BAI=",bai)
cai=angle((C-A),(I-A))
print("CAI",cai)

ABI=angle((A-B),(I-B))
print("ABI=",ABI)
cBi=angle((C-B),(I-B))
print("CBI=",cBi)

bCi=angle((B-C),(I-C))
print("BCI",bCi)
ACi=angle((A-C),(I-C))
print("ACI=",ACi)

#1.5.4
print("\n1.5.4")
k1=1
k2=1

p = np.zeros(2)
n1 = n_BC/ np.linalg.norm(n_BC)
n2 = n_CA / np.linalg.norm(n_CA)
n3 = n_AB / np.linalg.norm(n_AB)

p[0] = n1 @ B - k1 * n2 @ C
p[1] = n2 @ C - k2 * n3 @ A

N = np.block([[n1 - k1 * n2],[ n2 - k2 * n3]])
I = np.linalg.inv(N)@p
r = n1 @ (B-I)

print(f"Distance from I to BC= {r}")
print("\n1.5.5")
print("ID3=IE3=IF3=Inradius=r=",r)

#1.5.7 Draw circle with centre I

#1.5.8
print("\n1.5.8")
p=pow(np.linalg.norm(C-B),2)
q=2*((C-B)@(I-B))
s=pow(np.linalg.norm(I-B),2)-r*r

Discre=q*q-4*p*s

print("the Value of discriminant is ",abs(round(Discre,6)))
#  so the value of Discrimant is extremely small and tends to zero
#  the discriminant value rounded off to 6 decimal places is also zero
#  so it proves that there is only one solution of the point

#  the value of point is x=B+k(C-B)
k=((I-B)@(C-B))/((C-B)@(C-B))
print("the value of parameter k is ",k)
D_3=B+k*(C-B)
print("the point of tangency of incircle to side BC, D_3 is ",D_3)
#  to check we also check the value of dot product of ID3 and BC
#print("the dot product of ID3 and BC",abs(round(((D3-I)@(C-B),6))))
#  so this comes out to be zero

#1.5.9
print("\n1.5.9")
k2=((I-A)@(B-A))/((B-A)@(B-A))
F_3=A+k2*(B-A)
print("F_3=",F_3)

k3=((I-C)@(A-C))/((C-A)@(C-A))
E_3=C+k3*(A-C)
print("E_3=",E_3)

print("\n1.5.10")
print("AE_3=", np.linalg.norm(A -E_3) ,"\nAF_3=", np.linalg.norm(A-F_3) ,"\nBD_3=", np.linalg.norm(B-D_3) ,"\nBF_3=", np.linalg.norm(B-F_3) ,"\nCD_3=", np.linalg.norm(C-D_3) ,"\nCE_3=",np.linalg.norm(C-E_3))


#1.5.11
print("\n1.5.11")
a=BC
b=CA
c=AB
#creating array containing coefficients
Y = np.array([[1,1,0],[0,1,1],[1,0,1]])

#solving the equations
X = np.linalg.solve(Y,[c,a,b])

#printing output 
print(X)



