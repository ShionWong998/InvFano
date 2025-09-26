import numpy as np
input = np.genfromtxt('shape.csv', delimiter=',')
output = np.genfromtxt('result.csv', delimiter=',')
data_all = np.hstack((input, output))
names = 'x1,x2,phi1,phi2,alpha,ns,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
with open('data.csv', 'w') as f:
    f.write(names + '\n')
    np.savetxt(f, data_all, delimiter=',', fmt='%.5f')
