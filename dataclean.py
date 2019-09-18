import pandas as pd
import re
import os
import bert


BERTPATH='~/Models/chinese_wwm_ext_L-12_H-768_A-12/'
DATAPATH='~/DataSets/df1/'
BERTPATH=os.path.expanduser(BERTPATH)
DATAPATH=os.path.expanduser(DATAPATH)
OUTPUTPATH='./data/'
DATASETS=['DEV_Train_DataSet.csv']
DATASET_LABELS=['DEV_Train_DataSet_Label.csv']
datas=[]
answer=pd.read_csv(DATAPATH+DATASET_LABELS[0],index_col=0)
ans_dict=answer.to_dict()['label']
for dataset in DATASETS:
    data=pd.read_csv(DATAPATH+dataset)
    for line in data.shape[0]:
        sentence_id=data['id'][line]
        sentence_list=[]
        sentence_list.append(data['title'][line])
        content_fragment=re.split('([！？。])',data['content'][line])
        for i in range(0,len(content_fragment)-1,2):
            sentence_sym=content_fragment[i]+content_fragment[i+1]
            sentence_sym_c=re.sub('[\u0020-\u007e]{5,}','',sentence_sym)
            sentence_list.append(sentence_sym_c)
        with open(OUTPUTPATH+'OUTPUT_'+dataset,'w') as output:
            text='id,sentence,label\n'
            for sentence in sentence_list:
                text=text+'%s,%s,%d\n'%(sentence_id,sentence,ans_dict[sentence_id])
                # TODO tran sentence to vec by using bert
            output.write(text)
