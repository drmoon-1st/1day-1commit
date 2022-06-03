import pandas as pd
import numpy as np
a = [['a','b','c','a','b','a','c','a',],['d','b','c','d','b','d','c','d',],['d','b','c','a','b','a','c','a',]]
data = pd.DataFrame(np.array(a).T)
def multycode(data, idx):
    for i in idx:
        data[i] = data[i].astype('category').cat.codes
arr = [0, 1, 2]
multycode(data, arr)
print(data)