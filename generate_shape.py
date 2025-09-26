import numpy as np

r = np.array([3, 4.125, 5.25, 6.375, 7.5, 8.625, 9.5, 10.875, 12])
phi = np.array([10, 32.5, 55, 77.5, 100])
alpha = np.array([0.262, 0.393, 0.524, 0.654, 0.785])
ns = np.array([0.01, 0.2, 0.5, 0.8, 1])
structure = []

for ns1 in ns:
    for a in alpha:
        for p2 in phi:
            for p1 in phi:
                for r2 in r:
                    for r1 in r:
                        structure.append([r1, r2, p1, p2, a, ns1])

np.savetxt('shape.csv', structure, fmt='%3.3f', delimiter=',', newline='\n')