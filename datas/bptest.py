import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FowardSet(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.inputs = self.data.iloc[:, :6].values
        self.outputs = self.data.iloc[:, 6:].values
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,index):
        input_data = torch.tensor(self.inputs[index], dtype=torch.float32)
        output_data = torch.tensor(self.outputs[index], dtype=torch.float32)
        return input_data, output_data

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
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()


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



fnpath = 'fn.pt'
net = torch.load(fnpath)
net.to(device)

test0 = torch.tensor([0.263111111,0.690333333,0.464266667,0.264188889,0.585,0.01])
test1 = torch.tensor([0.263111111,0.690333333,0.464266667,0.264188889,0.585,0.5])
test2 = torch.tensor([0.263111111,0.690333333,0.464266667,0.264188889,0.585,1])


tt = torch.stack((test0, test1, test2), dim=0)
tt = tt.to(device)
outout = net(tt)
out_np = outout.cpu().detach().numpy()
with open ('bptest1.csv', 'ab') as f:
    np.savetxt(f,out_np, delimiter=',')
#print(out_np)

xx = np.linspace(3,5,201)
fig, ax = plt.subplots()
ax.plot(xx, out_np[0])
ax.plot(xx, out_np[1])
ax.plot(xx, out_np[2])
plt.show()