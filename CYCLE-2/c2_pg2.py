import numpy as np
x = np.array([[2+3j, 4+1j, 6+2j], [6+2j, 8+4j, 10+2j]])
print(type(x))	
print(x.shape)
print(x.dtype)
print(x)
print(x.reshape(3,2))