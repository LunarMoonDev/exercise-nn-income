import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy import genfromtxt
from sklearn.utils import shuffle

from model import TabularModel

# constants
CAT_TRAIN = './data/processed/cat_train.csv'
CON_TRAIN = './data/processed/con_train.csv'
Y_TRAIN = './data/processed/y_train.csv'
EMB_SZS = [(2, 1), (14, 7), (6, 3), (5, 3), (12, 6)]
VERSION = '0.0'

# data prep
cat_train = genfromtxt(CAT_TRAIN, delimiter = ',', dtype = np.float)
con_train = genfromtxt(CON_TRAIN, delimiter = ',', dtype = np.float)
y_train = genfromtxt(Y_TRAIN, delimiter = ',', dtype = np.float)

cat_train = torch.tensor(cat_train, dtype = torch.int64)
con_train = torch.tensor(con_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.long)

# print(cat_train)
# print(con_train)
# print(y_train)

# prep model
torch.manual_seed(33)
model = TabularModel(EMB_SZS, 2, 2, [50], p = 0.4)

# prep optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

start_time = time.time()
epochs = 300
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    if i % 25 == 1:
        print(f'epoch: {i: 3} loss: {loss.item(): 10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# saving the model
torch.save(model.state_dict(), f'./models/model.M.{VERSION}.{int(time.time())}.pt')
losses_np = np.array(losses)
pd.DataFrame(losses_np).to_csv('./data/processed/losses.csv', header = None, index = False)