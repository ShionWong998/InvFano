import S4 as S4
import numpy as np
import matplotlib.pyplot as plt
from findeps import findeps as findeps
#import time

#start = time.time()
c_const = 3e8
fmin = 3.5
fmax = 5.5
step = int((fmax-fmin)/0.01 + 1)
f1 = np.linspace(fmin, fmax, step)
f = f1 * 1e12
f0 = f / c_const * 1e-6

P = 30
R = 6
H = 13
t = 1e-3

names = ['0.01.csv','1.csv']

results = np.empty((0,len(f0)))
for i, name in enumerate(names):
    Rte = np.zeros(len(f0))
    Tte = np.zeros(len(f0))
    for ii, fi in enumerate(f0):
        ff = str(round(fi * c_const * 1e-6, 3))
        bpx, bpy, bpz = findeps(name, ff)
        S = S4.New(((P, 0), (0, P)), 25)
        S.SetMaterial('Air', 1.)
        S.SetMaterial('Si', 3.42 ** 2)
        S.SetMaterial('BP', ((bpx, 0, 0), (0, bpy, 0), (0, 0, bpz)))
        #S.SetMaterial('Mira',-1e10)

        S.AddLayer('AirTop', 0, 'Air')
        S.AddLayer('MonoBP', t, 'BP')
        S.AddLayer('Slab', H, 'Si')
        #S.AddLayer('Mira',1,'Mira')
        S.AddLayer('AirBot', 0, 'Air')

        S.SetRegionCircle('Slab', 'Air', (0,0), R)


        S.SetExcitationPlanewave((0, 0), 1, 0)


        S.SetFrequency(fi)
        inc, r = S.GetPowerFlux('AirTop', 0)
        fw, _ = S.GetPowerFlux('AirBot', 0)
        Rte[ii] = abs(-r / inc)
        Tte[ii] = abs(fw / inc)
        print(str(i)+','+str(ff))
    results = np.vstack((results,Tte))
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(f1, results[0],'r')
ax.plot(f1,results[1],'g')
#ax.plot(f1,results[2],'b')
plt.ylim([0,1])
#plt.savefig('b2.png')
fig.show()
#end = time.time()
#print(end-start)
