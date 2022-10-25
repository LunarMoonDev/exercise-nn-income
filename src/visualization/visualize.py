# import libraries
import numpy as np
import torch
from numpy import genfromtxt
import matplotlib.pyplot as plt

# constants
LOSS = './data/processed/losses.csv'

# prep the data
losses = genfromtxt(LOSS, delimiter = ',', dtype = np.float)
losses = torch.tensor(losses, dtype = torch.float)

plt.plot(range(300), losses.tolist())
plt.ylabel('RMSE Loss')
plt.xlabel('epoch')
plt.savefig('./reports/figures/loss_chart.png')
