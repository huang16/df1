import pandas as pd
import pickle as cPickle
import numpy as np
import os

PREDICT='~/Datasets/1t/df1/article1572169839'

SUBMITPATH='~/Datasets/df1/submit_example.csv'
PREDICT=os.path.expanduser(PREDICT)
SUBMITPATH=os.path.expanduser(SUBMITPATH)

raw=pd.read_csv(SUBMITPATH)
with open(PREDICT,'rb') as preptr:
    pre = cPickle.load(preptr)
for i in range(len(pre)):
    raw['label'][i]=np.argmax(pre[i])
    if i%1000==0:
        print(i)
raw.to_csv(SUBMITPATH,index=False,encoding='utf-8')
