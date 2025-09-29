import os
import csv
import numpy as np
import matplotlib.pyplot as plt

data_all = np.genfromtxt('wash_test.csv', delimiter=',')
#label_all = np.genfromtxt('sturcture.csv', delimiter=',')
abc = np.genfromtxt('rt.csv', delimiter=',')
data = data_all[1:,6:]
abc1 = abc[:,:]
rows = data.shape[0]
xx = np.linspace(3,5,201)
for i in range(rows):

    pic_name = './plots/' + str(i) + '.png'

    fig, ax = plt.subplots()
    ax.plot(xx, data[i], 'r')
    ax.plot(xx, abc1[i], 'b')
    plt.ylim([0,1])
    plt.savefig(pic_name)
    plt.close(fig)
    print(i)