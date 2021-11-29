num1= int(input(" enter the first number: "))
num2= int(input("enter the second number: "))
mn = min(num1, num2)
for i in range(1, mn+1):
 if num1%i==0 and num2%i==0:
    hcf = i
if hcf == 1:
 print("Entered numbers are coprime")
else:
 print(" They are not Co-Prime")