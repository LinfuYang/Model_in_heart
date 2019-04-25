import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# data = pd.read_table('C:/Users/Administrator/IdeaProjects/fitst_spqrk_pro/result.txt', sep =',', index_col=False)
data = []
for line in open('C:/Users/Administrator/IdeaProjects/fitst_spqrk_pro/result.txt'):
    list_1 = list(map(float, (line.strip()).split(',')))
    data.append(list_1)
data = np.array(data)
mse = np.sqrt(mean_squared_error(data[:, 0], data[:, 1]))
print(mse)