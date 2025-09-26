import S4 as S4
import numpy as np
import time
import math
import csv
import ast


start_time = time.time()
pi = math.pi
XY = np.empty((0, 2))

c_const = 3e8
fmin = 3
fmax = 5
step = int((fmax-fmin)/0.01 + 1)
f1 = np.linspace(fmin,fmax,step)
f = f1 * 1e12
f0 = f / c_const * 1e-6

filename = 'shape.csv'
P = 30  #period
tg = 1e-3 #BP thick 1nm

geom = np.zeros(8)
polygon = np.genfromtxt(filename, delimiter=',')

def findeps(filename,f):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == f:
                col2 = ast.literal_eval(row[1].strip())
                col3 = ast.literal_eval(row[2].strip())
                col4 = ast.literal_eval(row[3].strip())
                return col2,col3,col4
    return None,None,None

def Geom(r1, r2, Phi1, Phi2, alpha):
    Th1 = alpha * Phi1 / (Phi1 + Phi2)
    Th2 = alpha * Phi2 / (Phi1 + Phi2) + Th1

    Th = np.array([Th1,Th2])
    rr = np.array([r1,r2])
    Thi = np.array([])
    rii = np.array([])
    for i in range(8):
        Thi = np.hstack((Thi,Th+i*pi/4))
        rii = np.hstack((rii,rr))

    XY = np.empty((0,2))
    for j in range(len(Thi)):
        xx = rii[j] * np.cos(Thi[j])
        yy = rii[j] * np.sin(Thi[j])
        co = np.array([xx,yy])
        XY = np.vstack((XY,co))
    top = [tuple(row) for row in XY.tolist()]
    top1 = tuple(top)
    #print(top1)
    return top1

for pp, parameters in enumerate(polygon):

    geom = parameters
    coord = Geom(geom[0], geom[1], geom[2], geom[3], geom[4])
    #print(coord)
    thickness = 13
    ns = geom[5]
    ns1 = f'{ns:.2f}'
    BPpath = './BPeps/'
    epsname = BPpath + ns1 + '.csv'
    epsair = 1
    Trans = np.zeros(len(f0))
    S = S4.New(((P, 0), (0, P)), 25)
    S.SetOptions(
        Verbosity=0,  ## verbosity of the C++ code
        #DiscretizedEpsilon=True,  ## Necessary for high contrast
        DiscretizationResolution=8,  ## at least 8 if near field calculations
        LanczosSmoothing=True,  ## Mabe ?? especially for metals, for near fields
        SubpixelSmoothing=True,  ## definitely
        PolarizationDecomposition=True,  ## Along with 'normal', should help convergence
        PolarizationBasis='Normal'
    )
    S.SetMaterial('Air', epsair)
    S.SetMaterial('Si', 3.42 ** 2)
    bpx, bpy, bpz = 1, 1, 1
    S.SetMaterial('BP', ((bpx, 0, 0), (0, bpy, 0), (0, 0, bpz)))

    S.AddLayer('AirTop', 0, 'Air')
    S.AddLayer('MonoBP', tg, 'BP')
    S.AddLayer('Slab', thickness, 'Si')
    S.AddLayer('AirBot', 0, 'Air')

    S.SetRegionPolygon('Slab', 'Air', (0, 0), 0, coord)

    S.SetExcitationPlanewave((0, 0), 1, 0)  # te

    for ii, fi in enumerate(f0):
        ff = str(round(fi * c_const * 1e-6, 3))
        bpx, bpy, bpz = findeps(epsname, ff)
        S.SetMaterial('BP', ((bpx, 0, 0), (0, bpy, 0), (0, 0, bpz)))
        S.SetFrequency(fi)
        inc, _ = S.GetPowerFlux('AirTop', 0)
        fw, _ = S.GetPowerFlux('AirBot', 0)
        Trans[ii] = abs(fw / inc)
    result = Trans.reshape(1, len(f0))
    with open('result.csv', 'ab') as file:
        np.savetxt(file, result, delimiter=',', newline='\n', fmt='%.5f')

    print(str(pp) + 'is done.')

end_time = time.time()
dtime = end_time - start_time
mtime = dtime / 60
htime = (end_time - start_time) / 3600
print(str(dtime) + 'secs,')
if mtime > 1:
    print(str(mtime) + 'mins, ')
if htime > 1:
    print(str(htime) + 'hrs elapsed.')
print('This batch is end.')
