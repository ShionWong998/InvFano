import numpy as np
import math
from itertools import chain
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FowardNet(nn.Module):
    def __init__(self):
        super(FowardNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(6, 400))
        self.hidden_layers.append(nn.Linear(400, 600))
        for _ in range(3):
            self.hidden_layers.append(nn.Linear(600,600))
        self.hidden_layers.append(nn.Linear(600, 400))
        self.hidden_layers.append(nn.Linear(400, 201))

        self.output_layer = nn.Linear(201,201)

        self.LR = nn.LeakyReLU()



    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        x = self.relu(self.output_layer(x))

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.kaiming_normal_(m.bias.data)
fonet = torch.load('fn.pt')
fonet.to(device)

D = 6 #dimension
N_P = 200 #num of population
N_I = 3 #num of island
pi = math.pi
L = np.array([3,3,10,10,0.262,0.01])
U = np.array([12,12,100,100,0.785,0.1])

def create_population(L,U,N_P,N_I):
    pop = []
    for i in range(N_I):
        pop1 = []
        for j in range(N_P):
            sol = np.zeros(D)
            for k in range(D):
                sol[k] = L[k] + np.random.uniform() * (U[k] - L[k])
            pop1.append(sol)
        pop.append(pop1)
    return pop

def create_mutant(best, b, c, mut, L, U):
    mut_V = np.zeros(len(best))
    for i in range(len(best)):
        mut_V[i] = np.clip(best[i] + mut * (b[i] - c[i]), L[i], U[i])
    return mut_V

def crossover(mutant, curr, crossp):
    cross_points = np.random.rand(curr.size) > crossp
    return np.where(cross_points, mutant, curr)

def predict(all_geos, L, U):
    geos = np.copy(all_geos)
    for i in range(np.shape(geos)[0]):
        geos[i] = (geos[i] - L) / (U - L)
    with torch.no_grad():
        geo_tensor = torch.tensor(geos, dtype=torch.float32)
        geo_tensor = geo_tensor.to(device)
        spec_tensor = fonet(geo_tensor)
        spec_np = spec_tensor.cpu().detach().numpy()

    return spec_np



def get_mse(population, target, N_I, N_P, L, U):
    target = np.tile(target, (N_I * N_P, 1))
    island = list(chain.from_iterable(population))

    prediction = predict(island, L, U)
    m = np.mean((prediction - target)**2, axis = 1)
    return np.reshape(m, (N_I, N_P))


def de_island(fobj, #mse function
              target,
              L,
              U,
              num_layers_i, #dim
              num_islands,
              num_gens,
              poplist, #initialized population
              population_size,
              iters,
              verbose = 1,
              ):
    history = []
    bids = np.zeros(num_islands, dtype=int)
    bfits = np.zeros(num_islands)
    mutation_rate = np.random.uniform(0.4, 0.8, num_islands)
    cross_rate = np.random.uniform(0.5, 1.0, num_islands)
    num_func_evals = 0

    trilist = []
    for island in range(num_islands):
        cc = []
        for j in range(population_size):
            sol1 = np.zeros(num_layers_i)
            for k in range(num_layers_i):
                sol1[k] = L[k] + np.random.random() * (U[k] - L[k])
            cc.append(sol1)
        trilist.append(cc)

    tmp2 = np.random.uniform(0, 1, num_islands * num_layers_i)
    bests = np.split(tmp2, num_islands)

    for gen in range(num_gens):
        if verbose == 0:
            print('Epoch' + str(gen + 1))
        island = list(chain.from_iterable(poplist))
        fitness = fobj(poplist, target, num_islands, population_size, L, U)
        num_func_evals += len(island)

        for isln in range(num_islands):
            bids[isln] = np.argmin(fitness[isln])
            bests[isln] = poplist[isln][bids[isln]]

        for i in range(iters):
            for isln in range(num_islands):
                for j in range(population_size):
                    idxs = [idx for idx in range(population_size) if idx != j]
                    picks = np.random.choice(idxs, 3, replace = False)
                    a, b, c = poplist[isln][picks[0]], poplist[isln][picks[1]], poplist[isln][picks[2]]
                    mutant = create_mutant(a, b, c, mutation_rate[isln], L, U)
                    child = crossover(mutant, poplist[isln][j], cross_rate[isln])
                    trilist[isln][j] = child
            tflat = list(chain.from_iterable(trilist))
            f = fobj(trilist, target, num_islands, population_size, L, U)
            num_func_evals += len(tflat)

            for isln in range(num_islands):
                for j in range(population_size):
                    if f[isln][j] < fitness[isln][j]:
                        fitness[isln][j] = f[isln][j]
                        poplist[isln][j] = trilist[isln][j]
                        if f[isln][j] < fitness[isln][bids[isln]]:
                            bids[isln] = j
                            bests[isln] = trilist[isln][j]
                bfits[isln] = fitness[isln][bids[isln]]

            #if (i + 1) % 1 == 0:
                #print("Iteration = ", "%3d" %i, ' -- ')
            #with open('Data_bfits.txt', 'a') as data:
                #data.write(str(i) + '\t' + str(bfits) + '\n')
            #data.close()
            #with open('Data_best.txt', 'a') as data1:
                #np.savetxt(data1, bests, fmt='%4.3f')
            #data1.close()
            #print(bfits)
            history.append(np.copy(bfits))

        if iters>64:
            iters = int(iters / 2)

        if gen < (num_gens - 1):
            stmp = np.copy(poplist[num_islands-1][bids[num_islands-1]])
            for isln in range(num_islands-1, 0, -1):
                poplist[isln][bids[isln]] = np.copy(poplist[num_islands-1][bids[num_islands-1]])
            poplist[0][bids[0]] = stmp

    #print('Num func evals: ', num_func_evals)
    #print('Bid = ', bids, 'bests = ', bests, 'bfits = ', bfits, 'history = ', np.asarray(history))
    return bids, bests, bfits, np.asarray(history), num_func_evals


#target = np.array([[0.58667,0.59228,0.59797,0.60374,0.60958,0.61551,0.62151,0.62759,0.63374,0.63996,0.64626,0.65263,0.65907,0.66557,0.67214,0.67877,0.68547,0.69222,0.69902,0.70588,0.71278,0.71973,0.72671,0.73374,0.74079,0.74787,0.75498,0.7621,0.76923,0.77636,0.7835,0.79062,0.79774,0.80483,0.81189,0.81891,0.8259,0.83283,0.8397,0.8465,0.85322,0.85986,0.8664,0.87283,0.87915,0.88535,0.89141,0.89733,0.9031,0.9087,0.91414,0.91939,0.92445,0.92931,0.93396,0.9384,0.94261,0.94659,0.95032,0.95381,0.95705,0.96003,0.96274,0.96518,0.96736,0.96925,0.97087,0.97221,0.97326,0.97352,0.97456,0.9748,0.97477,0.97446,0.9739,0.97309,0.97204,0.97076,0.96926,0.96755,0.96566,0.96359,0.96137,0.95903,0.95659,0.95408,0.95153,0.949,0.94651,0.94413,0.94192,0.93996,0.93834,0.93718,0.9366,0.93662,0.93662,0.93621,0.94067,0.94447,0.94692,0.94218,0.90934,0.77755,0.41805,0.0443,0.08003,0.24802,0.36905,0.44461,0.49227,0.52314,0.5435,0.55696,0.56569,0.57107,0.574,0.57509,0.57476,0.57334,0.57104,0.56803,0.56446,0.56043,0.55601,0.55126,0.54625,0.54101,0.53557,0.52996,0.5242,0.51832,0.51232,0.5062,0.49999,0.49367,0.48726,0.48074,0.47412,0.46737,0.46049,0.45345,0.44624,0.43881,0.43114,0.42315,0.4148,0.40598,0.39654,0.38609,0.37634,0.36434,0.35061,0.33495,0.31651,0.29408,0.26575,0.22834,0.17663,0.10373,0.02142,0.14279,0.66455,0.7595,0.69264,0.63119,0.58628,0.55321,0.52798,0.50803,0.49176,0.47815,0.46652,0.45639,0.44745,0.43945,0.43221,0.42559,0.41945,0.41365,0.40778,0.39456,0.40189,0.39677,0.39251,0.38864,0.38503,0.38164,0.37845,0.37543,0.37257,0.36986,0.3673,0.36487,0.36257,0.3604,0.35834,0.3564,0.35458,0.35287,0.35128
#],[0.52253,0.52605,0.52963,0.53327,0.53697,0.54074,0.54458,0.54848,0.55244,0.55647,0.56057,0.56474,0.56897,0.57328,0.57765,0.58209,0.5866,0.59117,0.59582,0.60054,0.60532,0.61018,0.6151,0.6201,0.62516,0.6303,0.6355,0.64077,0.6461,0.65151,0.65698,0.66251,0.66812,0.67378,0.67951,0.68529,0.69114,0.69705,0.70301,0.70903,0.71509,0.72121,0.72738,0.73359,0.73985,0.74614,0.75248,0.75884,0.76523,0.77166,0.7781,0.78456,0.79103,0.79752,0.804,0.81049,0.81697,0.82344,0.82989,0.83632,0.84272,0.84908,0.8554,0.86168,0.86789,0.87405,0.88013,0.88613,0.89205,0.89788,0.9036,0.90922,0.91472,0.92009,0.92534,0.93044,0.93539,0.94019,0.94483,0.94929,0.95358,0.95768,0.96158,0.96529,0.96879,0.97207,0.97509,0.97743,0.98087,0.9832,0.98535,0.98728,0.98898,0.99044,0.99166,0.99265,0.9934,0.99391,0.99418,0.99423,0.99405,0.99365,0.99303,0.99221,0.99118,0.98996,0.98856,0.98699,0.98526,0.98339,0.98139,0.97927,0.97706,0.97477,0.97243,0.97006,0.96768,0.96533,0.96302,0.9608,0.9587,0.95676,0.95502,0.95354,0.95236,0.95155,0.95117,0.9513,0.95201,0.95339,0.95552,0.95847,0.96231,0.96705,0.97259,0.97861,0.98441,0.98845,0.98762,0.97586,0.94187,0.86691,0.72802,0.5187,0.28247,0.10006,0.01466,0.00671,0.03765,0.08093,0.12394,0.16206,0.19422,0.22069,0.24219,0.25948,0.27324,0.28403,0.29311,0.29888,0.30339,0.3064,0.30809,0.30863,0.30814,0.30671,0.30441,0.3013,0.29741,0.29276,0.28735,0.28115,0.27413,0.26623,0.25736,0.2474,0.23621,0.22357,0.20924,0.19286,0.174,0.15209,0.12622,0.1083,0.06581,0.03029,0.00313,0.01755,0.1618,0.52067,0.87103,0.96007,0.90299,0.82029,0.74732,0.68858,0.64186,0.60432,0.57364,0.54812,0.52653
 #   ]])
target = np.genfromtxt('./testset/rest.csv',delimiter=',')



for tn, target1 in enumerate(target):
    print(tn)
    population = create_population(L, U, N_P, N_I)
    (_,best1,fit1,_,_) = de_island(get_mse, target1, L, U, num_layers_i=D, num_islands=N_I, num_gens=5, poplist=population, population_size=N_P, iters=1000)
    ind1 = np.argmin(fit1)
    bb1 = best1[ind1]
    with open('outout.csv', 'ab') as file:
        np.savetxt(file, [bb1], delimiter=',', newline='\n', fmt='%.5f')
    print(tn)