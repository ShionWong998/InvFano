import numpy as np
import math
pi = math.pi
XY = np.empty((0, 2))

L = np.array([3,3,10,10,0,0])
U = np.array([12,12,100,100,1,1])
washed1 = np.array([0.651888889,0.107,0.711266667,0.081733333,0.444,0.427586207])
washed2 = np.array([0.263111111,0.690333333,0.464266667,0.264188889,0.585,0.093103448])
washed3 = np.array([0.728333333,0.734666667,0.513111111,0.188066667,0.598,0.224137931])
washed4 = np.array([0.071888889,0.159888889,0.037755556,0.325722222,0.411,0.544827586])

bp1 = np.array([0.651888889,0.107,0.711266667,0.081733333,0.444,0.01])
bp2 = np.array([0.263111111,0.690333333,0.464266667,0.264188889,0.585,0.01])
bp3 = np.array([0.634888888888889,0.8697777777777778,0.5640222222222222,0.3059555555555556,0.412,0.017])
bp4 = np.array([0.7689999999999999,0.14044444444444448,0.24564444444444442,0.5023555555555556,0.544,0.868])

inv1 = np.array([8.415,4.331,74.819,23.048,0.554,0.01])
inv2 = np.array([8.857,8.302,25.963,73.240,0.597,0.013])
x = (bp2* (U - L) + L)
#x = inv2
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

top1 = Geom(x[0], x[1], x[2], x[3], x[4])
print(top1)
print(x[5])
print(x)