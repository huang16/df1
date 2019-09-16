import pandas as pd
import re


DATAPATH='./data/'
DATASETS=[]
datas=[]
for dataset in DATASETS:
    data=pd.read_csv(DATAPATH+dataset)
    for content in data['content']:
        sentences=re.split('([！？。])',content)
    datas.append(data)