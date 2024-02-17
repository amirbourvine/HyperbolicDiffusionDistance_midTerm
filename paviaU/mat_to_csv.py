import scipy.io
import pandas as pd

mat = scipy.io.loadmat('PaviaU_gt.mat')
data_1 = mat['paviaU_gt']
df = pd.DataFrame(data_1)
df.to_csv("PaviaU_gt.csv")