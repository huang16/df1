import pandas as pd
import re
import os
import math
import bert

# TODO fix bug in 15c9756a7e2a4b6ca259c9bf9f748112, pandas can not split currently
# should split one line, but 4, 4ec089b73a73429c86c119830254bed6
class DataExample(object):
    def __init__(self,id,sentence,label=None):
        self.id=id
        self.sentence=sentence
        self.label=label
BERTPATH='~/Models/chinese_wwm_ext_L-12_H-768_A-12/'
DATAPATH='~/DataSets/df1/'
BERTPATH=os.path.expanduser(BERTPATH)
DATAPATH=os.path.expanduser(DATAPATH)
OUTPUTPATH='./data/'
DATASETS=['Train_DataSet.csv']
DATASET_LABELS=['Train_DataSet_Label.csv']
datas=[]
answer=pd.read_csv(DATAPATH+DATASET_LABELS[0],index_col=0)
ans_dict=answer.to_dict()['label']
for dataset in DATASETS:
    data=pd.read_csv(DATAPATH+dataset)
    sentence_list = []
    for line in range(data.shape[0]):
        sentence_id=data['id'][line]
        # TODO label is None in test set
        try:
            label=ans_dict[sentence_id]
        except KeyError:
            continue
        title=data['title'][line]
        if type(title) is str:
            if len(title)>1:
                sentence_list.append(DataExample(sentence_id,title,label))
        content=data['content'][line]
        if type(content) is str:
            content_fragment=re.split('([！？。])',content)
            if len(content_fragment)==0:
                continue
            if len(content_fragment)==1:
                sentence_list.append(DataExample(sentence_id, content, label))
            else:
                for i in range(0,len(content_fragment)-1,2):
                    sentence_sym=content_fragment[i]+content_fragment[i+1]
                    sentence_sym_c=re.sub('[\u0020-\u007e]{5,}','',sentence_sym)
                    sentence_list.append(DataExample(sentence_id,sentence_sym_c,label))
    with open(OUTPUTPATH+'OUTPUT_'+dataset,'w',encoding='utf-8') as output:
        text='id,sentence,label\n'
        for sentence in sentence_list:
            # TODO fix None in field sentence and label
            if type(sentence.sentence) is not str or len(sentence.sentence)==0:
                print(sentence.id)

            text=text+'%s,%s,%d\n'%(sentence.id,sentence.sentence,sentence.label)
            # TODO tran sentence to vec by using bert
        output.write(text)
