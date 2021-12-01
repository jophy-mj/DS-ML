import numpy as np
x=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])
print(x)
print("Display all elements excluding the first row")

print(x[1:4,:])

print("Display all elements excluding the last column")

print(x[:,0:3])

print("Display the elements of 1 st and 2 nd column in 2 nd and 3 rd row")

print(x[1:3,0:2])

print("Display the elements of 2 nd and 3 rd column")

print(x[:,1:3])

print("Display 2 nd and 3 rd element of 1 st row")

print(x[0,1:3])