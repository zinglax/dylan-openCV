import numpy as np

a = np.matrix("1,2,1;0,0,0;-1,-2,1")

m1 = np.matrix("5,5,5;1,1,1;7,7,7")
m2 = np.matrix("5,5,5;1,1,1;7,7,6")
m3 = np.matrix("5,5,5;1,1,5;7,6,5")

m4 = np.matrix("1,1,1;7,7,7;5,7,10")
m5 = np.matrix("1,1,1;7,7,6;7,10,12")
m6 = np.matrix("1,1,5;7,6,5;10,12,5")

m7 = np.matrix("7,7,7;5,7,10;5,7,7")
m8 = np.matrix("7,7,6;7,10,12;7,7,5")
m9 = np.matrix("7,6,5;10,12,5;7,5,5")


print m1
print np.dot(m1, a)

#n = 1/9
#nineths = n + "," + n
#nineth = np.matrix(",")
#one = np.matrix("0,0,0;0,1,0;0,0,0")
#print nineth
#print np.dot(nineth,one)