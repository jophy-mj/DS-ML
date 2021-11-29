lt1 = [5, 10, 15, 20, 25, 30]
lt2 = [2, 4, 6, 8, 10, 12]
print ( " Python Original list 1: " + str (lt1))
print ( "Python Original list 2: " + str (lt2))
res_lt = []
for x in range (0, len (lt1)):
    res_lt.append( lt1[x] + lt2[x])
print ( " Addition of the list lt1 and lt2 is: " + str (res_lt))