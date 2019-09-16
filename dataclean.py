import pandas as pd
import re


DATAPATH='./data/'
DATASETS=[]
DATASET_LABELS=[]
datas=[]
for dataset in DATASETS:
    data=pd.read_csv(DATAPATH+dataset)
    for line in data.shape[0]:
        sentence_id=data['id'][line]
        sentence_list=[]
        sentence_list.append(data['title'][line])
        content_fragment=re.split('([！？。])',data['content'])
        for i in range(0,len(content_fragment)-1,2):
            sentence_list.append(content_fragment[i]+content_fragment[i+1])
        # TODO clean jscode in the content
        with open(DATAPATH+'OUTPUT_'+dataset,'w') as output:
            text='id,sentence\n'
            for sentence in sentence_list:
                text=text+'%s,%s\n'%(sentence_id,sentence)
                # TODO tran sentence to vec by using bert
            output.write(text)
