import numpy as np

part_all = np.empty((0,207))
for i in range(10):
    part_name = str(i+1) + '/data.csv'
    part = np.genfromtxt(part_name, delimiter=',')
    part_all = np.vstack((part_all, part))
    print(i+1)
np.savetxt('data_all.csv', part_all, delimiter=',', fmt='%.5f')