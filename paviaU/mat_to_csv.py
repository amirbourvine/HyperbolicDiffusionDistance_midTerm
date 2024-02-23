import scipy.io
import pandas as pd

mat = scipy.io.loadmat('paviaU/PaviaU.mat')
data_1 = mat['paviaU']
df = pd.DataFrame(data_1)
df.to_csv("PaviaU_try.csv")