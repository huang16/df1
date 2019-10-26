import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])

import pandas as pd
import re
import os
from extract_features import ExtractConf,extract
import pickle as cPickle
from sklearn.model_selection import train_test_split


class DataExample(object):
    def __init__(self, id, sentence, label=None):
        self.id = id
        self.sentence = sentence
        self.label = label

    __setitem__ = object.__setattr__
    __getitem__ = object.__getattribute__


BERTPATH = '~/Models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/'
DATAPATH = '~/DataSets/df1/'
OUTPUTPATH = '~/DataSets/1t/df1/'
BERTPATH = os.path.expanduser(BERTPATH)
DATAPATH = os.path.expanduser(DATAPATH)
OUTPUTPATH=os.path.expanduser(OUTPUTPATH)
DATASETS = ['Train_DataSet.csv','Test_DataSet.csv']
DATASET_LABELS = 'Train_DataSet_Label.csv'
EVALSET=True
# 7354:764,3659,2932
datas = []

def load_answers():
    answer = pd.read_csv(DATAPATH + DATASET_LABELS, index_col=0)
    ans_dict = answer.to_dict()['label']
    return ans_dict
def read_raw_dataset(dataset,evalSet=False):
    ans_dict=load_answers()
    with open(DATAPATH + dataset,encoding='utf-8') as dataset_file:
        raw_data = dataset_file.readlines()[1:]
    if not evalSet:
        raw_data_tat=train_test_split(raw_data,test_size=0.2,random_state=42)
    else:
        raw_data_tat =[raw_data]
    convert_list_r=[]
    for raw_data in raw_data_tat:
        raw_data_split = []
        index = 0
        for i in raw_data:
            index += 1
            split = i.split(',')
            if len(split) == 1:
                print('ERR::fragment %s, in line %d' % (str(i), index - 1))
                continue
            if len(split) == 2:
                print('WARN::fragment %s, in line %d' % (str(i), index - 1))
                split.append('')
            raw_data_split.append(DataExample(split[0], split[1], split[2]))
        sentence_list = []
        index = 0
        for line in raw_data_split:
            sentence_subid=0
            sentence_id = line.id
            index += 1
            try:
                label = ans_dict[sentence_id]
            except KeyError:
                label = -1
                print('ERR::invalid id %s, in line %d' % (str(sentence_id), index - 1))
                if not re.match('Test', dataset):
                    continue
            title = line.sentence
            if type(title) is str and len(title) > 0:
                sentence_in = re.sub(r'(\s)|[\u0020-\u007e]{5,}', '', title)
                while len(sentence_in) > 128:
                    sentence_list.append(DataExample(sentence_id+'%d'%sentence_subid, sentence_in[:128], label))
                    sentence_subid+=1
                    sentence_in = sentence_in[128:]
                sentence_list.append(DataExample(sentence_id+'%d'%sentence_subid, sentence_in, label))
                sentence_subid += 1
            else:
                print('ERR::invalid title %s, in line %d' % (str(title), index - 1))
            content = line.label
            if type(content) is str and len(content) > 0:
                content_fragment = re.split('([！!？?。;；、])', content)
                if len(content_fragment) == 0:
                    continue
                if len(content_fragment) == 1:
                    sentence_in = re.sub(r'(\s)|[\u0020-\u007e]{5,}', '', content)
                    while len(sentence_in) > 128:
                        sentence_list.append(DataExample(sentence_id+'%d'%sentence_subid, sentence_in[:128], label))
                        sentence_subid += 1
                        sentence_in = sentence_in[128:]
                    sentence_list.append(DataExample(sentence_id+'%d'%sentence_subid, sentence_in, label))
                    sentence_subid += 1
                else:
                    for i in range(0, len(content_fragment) - 1, 2):
                        sentence_sym = content_fragment[i] + content_fragment[i + 1]
                        sentence_in = re.sub(r'(\s)|[\u0020-\u007e]{5,}', '', sentence_sym)
                        while len(sentence_in) > 128:
                            sentence_list.append(DataExample(sentence_id+'%d'%sentence_subid, sentence_in[:128], label))
                            sentence_subid += 1
                            sentence_in = sentence_in[128:]
                        sentence_list.append(DataExample(sentence_id+'%d'%sentence_subid, sentence_in, label))
                        sentence_subid += 1
            else:
                print('ERR::invalid content %s, in line %d' % (str(content), index - 1))
        convert_list = []
        with open(OUTPUTPATH + 'OUTPUT_' + dataset, 'wb') as output:
            print('dump path:%s'%output.name)
            text = 'id,sentence,label\n'
            for sentence in sentence_list:
                if type(sentence.sentence) is not str or len(sentence.sentence) == 0:
                    log = 'frag data: %s,%s,%s' % (str(sentence.id), str(sentence.sentence), str(sentence.label))
                    print(log)
                elif not re.match('[\u0030-\u0039\u0061-\u007a]{32}', sentence.id):
                    log = 'wrong data: %s,%s,%s' % (str(sentence.id), str(sentence.sentence), str(sentence.label))
                    print(log)
                else:
                    text = text + '%s,%s,%d\n' % (sentence.id, sentence.sentence, sentence.label)
                    convert_list.append(sentence)
            cPickle.dump(convert_list,output)
        print('len convert_list:  %d' % (len(convert_list)))
        os.rename(OUTPUTPATH + 'OUTPUT_' + dataset,OUTPUTPATH + 'OUTPUT_' + dataset+'%d'%len(convert_list))
        convert_list_r.append(convert_list)
    return convert_list_r, ans_dict


if __name__ == '__main__':
    # TODO use pickle to save clean_data and reload clean_data
    if EVALSET:
        if os.path.exists(OUTPUTPATH + 'OUTPUT_' + DATASETS[1]+'280339',):
            convert_list=[cPickle.load(open(OUTPUTPATH + 'OUTPUT_' + DATASETS[1]+'280339',mode='rb'))]
            ans_dict=load_answers()
        else:
            (convert_list,ans_dict)=read_raw_dataset(DATASETS[1],evalSet=EVALSET)
    else:
        if os.path.exists(OUTPUTPATH + 'OUTPUT_' + DATASETS[0],):
            convert_list=[cPickle.load(open(OUTPUTPATH + 'OUTPUT_' + DATASETS[0],mode='rb'))]
            ans_dict=load_answers()
        else:
            (convert_list,ans_dict)=read_raw_dataset(DATASETS[0],evalSet=EVALSET)
    #convert_list_train, convert_list_test = train_test_split(convert_list, test_size=0.2, random_state=42)
    for convert_list_item in convert_list:
        config=ExtractConf(sentence_field='sentence',
                           id_field='id',
                           input_list=convert_list_item,
                           label_dict=ans_dict,
                           batch_size=96,
                           bert_folder=BERTPATH,
                           output_folder=OUTPUTPATH)
        extract(config)