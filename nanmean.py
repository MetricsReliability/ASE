import pandas as pd

data = pd.read_csv('auc.csv', sep=',')
data.fillna(data.mean(), inplace=True)
data.to_csv('new_auc.csv', index=False)
