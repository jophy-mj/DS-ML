a=int(input("Enter start index:"))
b=int(input("Enter end index:"))
for x in range(a,b):
    if x > 1:
        for i in range(2,x):
          if (i%2==0):
          print(x)

