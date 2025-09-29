import S4 as S4
import numpy as np
import matplotlib.pyplot as plt
from findeps import findeps as findeps
#import time

#start = time.time()
c_const = 3e8
fmin = 3
fmax = 5
step = int((fmax-fmin)/0.01 + 1)
f1 = np.linspace(fmin, fmax, step)
f = f1 * 1e12
f0 = f / c_const * 1e-6

P = 30
Rlist = np.linspace(0,12,101)
H = 13
t = 1e-3

name = '0.01.csv'

results = np.empty((0,len(f0)))
Rte = np.zeros(len(f0))
Tte = np.zeros(len(f0))
for R in Rlist:
    for ii, fi in enumerate(f0):
        ff = str(round(fi * c_const * 1e-6, 3))
        bpx, bpy, bpz = findeps(name, ff)
        S = S4.New(((P, 0), (0, P)), 25)
        S.SetMaterial('Air', 1.)
        S.SetMaterial('Si', 3.42 ** 2)
        S.SetMaterial('BP', ((bpx, 0, 0), (0, bpy, 0), (0, 0, bpz)))
        #S.SetMaterial('Mira',-1e10)

        S.AddLayer('AirTop', 0, 'Air')
        #S.AddLayer('MonoBP', t, 'BP')
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
    result = Tte.reshape(1, len(f0))
    with open('result.csv', 'ab') as file:
        np.savetxt(file, result, delimiter=',', newline='\n', fmt='%.5f')
    print(str(R))

#fig = plt.figure()
#ax = fig.add_subplot()
#ax.plot(f1, Tte, 'r')
#ax.plot(f1,results[2],'b')
#plt.ylim([0,1])
#plt.savefig('b2.png')
#fig.show()
#end = time.time()
#print(end-start)
#np.savetxt('the/'+str(R)+'.csv', Tte, delimiter=',', newline='\n', fmt='%.5f')