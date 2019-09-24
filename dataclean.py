import pandas as pd
import re
import os
from extract_features import ExtractConf,extract
import pickle as cPickle

# TODO refactor for multiplexing
class DataExample(object):
    def __init__(self, id, sentence, label=None):
        self.id = id
        self.sentence = sentence
        self.label = label

    __setitem__ = object.__setattr__
    __getitem__ = object.__getattribute__


BERTPATH = '~/Models/chinese_wwm_ext_L-12_H-768_A-12/'
DATAPATH = '~/DataSets/df1/'
BERTPATH = os.path.expanduser(BERTPATH)
DATAPATH = os.path.expanduser(DATAPATH)
OUTPUTPATH = './data/'
DATASETS = ['Train_DataSet.csv']
DATASET_LABELS = 'Train_DataSet_Label.csv'
datas = []

def load_answers():
    answer = pd.read_csv(DATAPATH + DATASET_LABELS, index_col=0)
    ans_dict = answer.to_dict()['label']
    return ans_dict
def read_raw_dataset(dataset):
    ans_dict=load_answers()
    with open(DATAPATH + dataset,encoding='utf-8') as dataset_file:
        raw_data = dataset_file.readlines()[1:]
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
        sentence_id = line.id
        index += 1
        # TODO label is None in test set
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
                sentence_list.append(DataExample(sentence_id, sentence_in[:128], label))
                sentence_in = sentence_in[128:]
            sentence_list.append(DataExample(sentence_id, sentence_in, label))
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
                    sentence_list.append(DataExample(sentence_id, sentence_in[:128], label))
                    sentence_in = sentence_in[128:]
                sentence_list.append(DataExample(sentence_id, sentence_in, label))
            else:
                for i in range(0, len(content_fragment) - 1, 2):
                    sentence_sym = content_fragment[i] + content_fragment[i + 1]
                    sentence_in = re.sub(r'(\s)|[\u0020-\u007e]{5,}', '', sentence_sym)
                    while len(sentence_in) > 128:
                        sentence_list.append(DataExample(sentence_id, sentence_in[:128], label))
                        sentence_in = sentence_in[128:]
                    sentence_list.append(DataExample(sentence_id, sentence_in, label))
        else:
            print('ERR::invalid content %s, in line %d' % (str(content), index - 1))
    convert_list = []
    with open(OUTPUTPATH + 'OUTPUT_' + dataset, 'w', encoding='utf-8') as output:
        text = 'id,sentence,label\n'
        for sentence in sentence_list:
            # TODO fix None in field sentence and label
            if type(sentence.sentence) is not str or len(sentence.sentence) == 0:
                log = 'frag data: %s,%s,%s' % (str(sentence.id), str(sentence.sentence), str(sentence.label))
                print(log)
            elif not re.match('[\u0030-\u0039\u0061-\u007a]{32}', sentence.id):
                log = 'wrong data: %s,%s,%s' % (str(sentence.id), str(sentence.sentence), str(sentence.label))
                print(log)
            else:
                text = text + '%s,%s,%d\n' % (sentence.id, sentence.sentence, sentence.label)
                convert_list.append(sentence)
            # TODO tran sentence to vec by using bert
        output.write(text)
    print('len convert_list:  %d' % (len(convert_list)))
    return convert_list, ans_dict


if __name__ == '__main__':
    # TODO use pickle to save clean_data and reload clean_data
    (convert_list,ans_dict)=read_raw_dataset(DATASETS[0])
    config=ExtractConf(sentence_field='sentence',
                       id_field='id',
                       input_list=convert_list,
                       label_dict=ans_dict,
                       batch_size=32)
    extract(config)
