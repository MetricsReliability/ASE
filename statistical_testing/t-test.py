## Import the packages
import numpy as np
from scipy import stats
import pandas as pd

ds_addr = "E:\\apply\\york\\project\\software\\statistical testing framework\\ESEM2020\\reca.csv"

_ds_ = pd.read_csv(filepath_or_buffer=ds_addr, index_col=None)

## Define 2 random distributions
# Sample Size
[m, N] = _ds_.shape
# Gaussian distributed data with mean = 2 and var = 1
a = _ds_.iloc[:, 2]
# Gaussian distributed data with with mean = 0 and var = 1
b = _ds_.iloc[:, 3]

## Calculate the Standard Deviation
# Calculate the variance to get the standard deviation

# For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

# std deviation
s = np.sqrt((var_a + var_b) / 2)

## Calculate the t-statistics
t = (a.mean() - b.mean()) / (s * np.sqrt(2 / m))

## Compare with the critical t-value
# Degrees of freedom
df = 2 * m - 2

# p-value after comparison with the t
p = 1 - stats.t.cdf(t, df=df)

print("t = " + str(t))
print("p = " + str(2 * p))
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(a, b)
print("t = " + str(t2))
print("p = " + str(p2))
