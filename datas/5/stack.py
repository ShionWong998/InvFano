import numpy as np

input = np.genfromtxt('1.csv', delimiter=',')
output = np.genfromtxt('result.csv', delimiter=',')
data_all = np.hstack((input, output))
np.savetxt('data.csv', data_all, delimiter=',', fmt='%.5f')