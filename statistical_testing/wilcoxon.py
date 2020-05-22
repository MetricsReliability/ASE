from scipy.stats import wilcoxon
import pandas as pd

ds_addr = "E:\\apply\\york\\project\\software\\statistical testing framework\\ESEM2020\\auc.csv"

_ds_ = pd.read_csv(filepath_or_buffer=ds_addr, index_col=None)

[m, N] = _ds_.shape

CKJM = _ds_.iloc[:, 1]
JMT = _ds_.iloc[:, 2]
Understand = _ds_.iloc[:, 3]

[statistics, pvalue] = wilcoxon(JMT, Understand)

print(pvalue)
print(statistics)
