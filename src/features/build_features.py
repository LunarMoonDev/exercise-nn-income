# import lirbaries
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle

# constants
INCOME = './data/raw/income.csv'

# util methods

# data prep
df = pd.read_csv(INCOME)
# print("content of income dataframe (head): \n", df.head())

cat_cols = ["sex", "education-num", "marital-status", "workclass", "occupation"]
con_cols = ["age", "hours-per-week"]
y_col = ["label"]

for cat in cat_cols:
    df[cat] = df[cat].astype('category')

df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
# print("content of income dataframe (head): \n", df.head())

cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size + 1) // 2)) for size in cat_szs]
print("Content of emb_szs: ", emb_szs)

cat_list = []
for col in cat_cols:
    cat_list.append(df[col].cat.codes.values)
cats = np.stack(cat_list, 1)
cats = torch.tensor(cats, dtype = torch.int64)

conts = np.stack([df[col].values for col in con_cols], 1)
conts = torch.tensor(conts, dtype = torch.float32)

y = torch.tensor(df[y_col].values, dtype = torch.float).reshape(-1, 1)

pd.DataFrame(cats.numpy()).to_csv('./data/interim/cats.csv', header = None, index = False)
pd.DataFrame(conts.numpy()).to_csv('./data/interim/conts.csv', header = None, index = False)
pd.DataFrame(y.numpy()).to_csv('./data/interim/y.csv', header = None, index = False)

b = 30000   # using this only
t = 500

cat_train = cats[:b - t]
cat_test = cats[b - t: b]
con_train = conts[:b - t]
con_test = conts[b - t: b]
y_train = y[: b - t]
y_test = y[b - t: b]

pd.DataFrame(cat_train.numpy()).to_csv('./data/processed/cat_train.csv', header = None, index = False)
pd.DataFrame(cat_test.numpy()).to_csv('./data/processed/cat_test.csv', header = None, index = False)
pd.DataFrame(con_train.numpy()).to_csv('./data/processed/con_train.csv', header = None, index = False)
pd.DataFrame(con_test.numpy()).to_csv('./data/processed/con_test.csv', header = None, index = False)
pd.DataFrame(y_train.numpy()).to_csv('./data/processed/y_train.csv', header = None, index = False)
pd.DataFrame(y_test.numpy()).to_csv('./data/processed/y_test.csv', header = None, index = False)
