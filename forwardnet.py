import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt


seed_value = 999
torch.manual_seed(seed_value)
seed = torch.initial_seed()
print(f'seed = {seed}')

#ltime = str(time.ctime()).replace(' ','_').replace(':','_')
#os.system('mkdir '+ltime)
#start_time = time.time()
#with open('./' + ltime + '/logs.txt', 'a') as log:
#    log.write(f'{seed} \n')
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


num_epochs = 2000

fonet = FowardNet()
#print(net)
#with open('./' + ltime + '/logs.txt', 'a') as log:
#    log.write(f'{net} \n')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
fonet.to(device)


#train_path = 'train1.pkl'
#with open(train_path, 'rb') as f1:
#    dataset = pickle.load(f1)

dataset_path = 'wash_all.csv'
dataset = FowardSet(dataset_path)
train_ratio = 0.9
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=True)




criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(fonet.parameters(), lr=1e-3, weight_decay=1e-6)


trloss = []
tsloss = []
for epoch in range(num_epochs):
    running_loss = 0.
    for batch_inputs, batch_outputs in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_outputs = batch_outputs.to(device)
        outputs = fonet(batch_inputs)

        loss = criterion(outputs, batch_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    trloss.append(train_loss)
    print(f'Epoch {epoch+1}/{num_epochs}: Loss = {train_loss:.6f}')
    #with open('./'+ltime+'/logs.txt', 'a') as log:
    #    log.write(f'Epoch {epoch+1}/{num_epochs}: Loss = {train_loss:.6f} \n')
    #if epoch % 100 == 0:
    #    torch.save(net, './'+ltime + '/net' + str(epoch) + '.pt')

    with torch.no_grad():
        err = 0.
        for batch_inputs, batch_outputs in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            outputs = fonet(batch_inputs)
            loss = criterion(outputs, batch_outputs)
            err += loss.item()
        test_loss = err / len(test_loader)
        tsloss.append(test_loss)
        print(f'test_loss: {test_loss:.6f}')
        #with open('./'+ltime+'/logs.txt', 'a') as log:
        #    log.write(f'test_loss: {err / len(test_loader):.6f} \n')
torch.save(fonet, 'fn.pt')
#np.savetxt('train.txt', trloss)
#np.savetxt('val.txt', tsloss)
#print('Finished Training')
#endtime = time.time()
#stime = round(endtime - start_time, 3)
#mtime = round(stime / 60, 1)
#htime = round(mtime / 60, 1)
#print(str(stime) + ' secs, ' + str(mtime) + ' mins, ' + str(htime) + 'hrs')
#print('done.')

xx = np.linspace(1, num_epochs+1, num_epochs)
fig, ax = plt.subplots()
ax.semilogy(xx, trloss, label='train', color='b')
ax.semilogy(xx, tsloss, label='test', color='r')
plt.savefig('loss.png')
fig.show()