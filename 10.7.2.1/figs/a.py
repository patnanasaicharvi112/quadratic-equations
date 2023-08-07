import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math 
import sys     #for path to external scripts 
  
def find_x(point1,point2,m,n):
    A=np.array(point1)
    B=np.array(point2)

    proj = ((m*B)+(n*A))/(m+n)
    return proj

#Two aray vectors are given
point1= np.array(([-1,7]))
point2= np.array(([4,-3]))
m=2
n=3
d=find_x(point1,point2,m,n)



A = np.array(([-1, 7]))
B = np.array(([4, -3]))
 
 
def line_gen(A,B): 
   len =10 
   dim = A.shape[0] 
   x_AB = np.zeros((dim,len)) 
   lam_1 = np.linspace(0,1,len) 
   for i in range(len): 
     temp1 = A + lam_1[i]*(B-A) 
     x_AB[:,i]= temp1.T 
   return x_AB 
 
   
x_AB = line_gen(A,B) 

 
 
 
#Plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$') 

 
 
 
#Labeling the coordinates 
tri_coords = np.vstack((A,B,d)).T 
plt.scatter(tri_coords[0,:], tri_coords[1,:]) 
vert_labels = ['A','B','D'] 
for i, txt in enumerate(vert_labels): 
    plt.annotate(txt, # this is the text 
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label 
                 textcoords="offset points", # how to position the text 
                 xytext=(0,10), # distance from text to points (x,y) 
                 ha='center') # horizontal alignment can be left, right or center 
 
plt.xlabel('$x-axis$') 
plt.ylabel('$y-axis$') 
plt.legend(loc='best') 
plt.grid() # minor 
plt.axis('equal') 
plt.title('Line segment AB',size=12) 

plt.show()
