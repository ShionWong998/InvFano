import numpy as np

r = np.array([3, 4.125, 5.25, 6.375, 7.5, 8.625, 9.5, 10.875, 12])
phi = np.array([10, 32.5, 55, 77.5, 100])
alpha = np.array([0.262, 0.393, 0.524, 0.654, 0.785])
ns = np.array([0.01, 0.03, 0.05, 0.07, 0.1])
structure = []
lower = np.array([3,3,10,10,0.262,0.01])
upper = np.array([12,12,100,100,0.785,1])
for i in range(2000):
    xx = np.zeros(6)
    for j in range(6):
        xx[j] = lower[j] + np.random.uniform(0,1) * (upper[j] - lower[j])
    structure.append(xx)
np.savetxt('test.csv', structure, fmt='%3.3f', delimiter=',', newline='\n')