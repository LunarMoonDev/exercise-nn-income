# import libraries
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy import genfromtxt
from model import TabularModel

# constants
CAT_TEST = './data/processed/cat_test.csv'
CON_TEST = './data/processed/con_test.csv'
Y_TEST = './data/processed/y_test.csv'
EMB_SZS = [(2, 1), (14, 7), (6, 3), (5, 3), (12, 6)]
MODEL_NAME = 'model.M.0.0.1665485811.pt'

# data prep
cat_test = genfromtxt(CAT_TEST, delimiter = ',', dtype = np.int64)
con_test = genfromtxt(CON_TEST, delimiter = ',', dtype = np.float)
y_test = genfromtxt(Y_TEST, delimiter = ',', dtype = np.float)

cat_test = torch.tensor(cat_test, dtype = torch.int64)
con_test = torch.tensor(con_test, dtype = torch.float)
y_test = torch.tensor(y_test, dtype = torch.long)

# prep model
torch.manual_seed(33)
model = TabularModel(EMB_SZS, 2, 2, [50], p = 0.4)
model.load_state_dict(torch.load(f'./models/{MODEL_NAME}'))
model.eval()

# prep criterion
criterion = nn.CrossEntropyLoss()

# evaluating
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))

print(f'RMSE: {loss: .8f}')

rows = 50
correct = 0
print(f'{"MODEL OUTPUT":26} ARGMAX  Y_TEST')
for i in range(rows):
    print(f'{str(y_val[i]):26} {y_val[i].argmax():^7}{y_test[i]:^7}')
    if y_val[i].argmax().item() == y_test[i]:
        correct += 1
print(f'\n{correct} out of {rows} = {100*correct/rows:.2f}% correct')