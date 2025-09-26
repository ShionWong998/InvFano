import numpy as np
import csv


def BPeps(fmin,fmax,step,ns):
    f1 = np.linspace(fmin,fmax,step)
    f = f1 * 1e12
    w = 2 * np.pi * f

    e = 1.602e-19
    m0 = 9.109e-31
    n = ns * 1e17
    eta = 1e-2 * e
    hbar = 6.626e-34 / 2 / np.pi
    eps0 = 8.854e-12
    epsr = 5.76

    gamma = 4 * 1e-9 / np.pi * e
    delta = 2 * e

    eta_c = hbar**2 / 0.4 / m0
    nu_c = hbar**2 / 0.7 / m0

    mcx = hbar**2 / (2 * gamma**2 / delta + eta_c)
    mcy = hbar**2 / nu_c

    dx = np.pi * e**2 * n / mcx
    dy = np.pi * e**2 * n / mcy

    sigx = 1j * dx / np.pi / (w + 1j * eta / hbar)
    sigy = 1j * dy / np.pi / (w + 1j * eta / hbar)

    epsx1 = epsr + 1j * sigx / eps0 / w / 1e-9
    epsy1 = epsr + 1j * sigy / eps0 / w / 1e-9
    epsz = epsr * np.ones(len(f1))
    epsx = np.round(epsx1,3)
    epsy = np.round(epsy1, 3)
    return epsx, epsy, epsz


#############

fmin = 3    #THz
fmax = 5    #THz
step = int((fmax-fmin)/0.01 + 1)    #+1

nslist = np.linspace(0.301,1,int((1-0.301)/0.001 + 1))
for ns in nslist:
    ns1 = f"{ns:.3f}"
    name = ns1 + '.csv'
    epsx, epsy, epsz = BPeps(fmin,fmax, step, ns)

    ff1 = np.linspace(fmin, fmax, step)
    ff = np.around(ff1,4)
    arrays = [ff, epsx, epsy, epsz]
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in zip(*arrays):
            writer.writerow(row)

    print(f'saved into {name}')



