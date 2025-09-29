import numpy as np

target = np.genfromtxt('rest1.csv',delimiter=',')
abc = np.genfromtxt('abc.csv',delimiter=',')

ddd = target-abc
ddd2 = ddd * ddd
ddd3 = np.sum(ddd2)
print(ddd3)
ddd4 = ddd3/(201*100)
print(ddd4)
#np.savetxt('ddd2.csv', ddd2, delimiter=',', newline='\n', fmt='%.5f')