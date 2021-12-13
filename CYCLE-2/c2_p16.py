import numpy as np
A=np.array([[2,3,6],[3,4,5],[6,5,9]])
B=np.array([[0,2,9],[-2,0,3],[-9,-3,0]])
AT=A.transpose()
comparison = A == AT
equal_arrays = comparison.all()
print(equal_arrays)
m=(B.transpose() == -B).all()
print(m)

  
#import numpy as np

#M = np.array([[2,3,4], [3,45,8], [4,8,78]])
#print('\n\nMatrix M\n', M)
# Transposing the Matrix M
#print('\n\nTranspose as M.T\n', M.T)

#if M.T.all() == M.all():
   # print("Matrix is symmetric")
#else:
  #  print("The given matrix is not symmetric")
#A=np.array([[1,3,7], [3,-5,8]])   
    
#if A.T.all() == ~A.all():
 #   print("matrix is skew symmetric")
#else:
  #  print("matrix is not skew matrix")