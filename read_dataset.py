import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
import tensorflow as tf
from tensorflow import keras
import os
import time
import pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt


DEFAULTCHECKPOINT='./checkpoint/savemodel'
DEFAULTPREDICT='./predict'
DEV=True
TRAIN=False
CHECKPOINT='./checkpoint/savemodel1572111169.h5'
PREDICT=True
'''
train_dataset=tf.data.TFRecordDataset(train_dataset_path)
data=train_dataset.take(1)
for raw_data in data:
    example=tf.train.Example()
    example.ParseFromString(raw_data.numpy())
    label=example.features.feature['label'].float_list.value[0]
    id=example.features.feature['id'].bytes_list.value[0]
    matrix=example.features.feature['matrix'].bytes_list.value[0]
    nd_matrix=np.frombuffer(matrix,dtype=np.float32).reshape((4,128,768))
    print(label)
    print(id)
print(data)
'''

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()



def parse_fn(example):

    example_fmt={
        'id': tf.io.FixedLenFeature([], tf.string),
        'matrix': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32)
    }

    features=tf.io.parse_single_example(example,example_fmt)
    id=features['id']
    matrix=features['matrix']
    label=features['label']
    matrix=tf.io.decode_raw(matrix,tf.float32)
    matrix = tf.reshape(matrix, (4, 128, 1024))
    matrix = tf.transpose(matrix, perm=[1, 2, 0])
    matrix = tf.reshape(matrix, (128,4096))
    label=tf.cast(label,dtype=tf.int32)
    label=tf.one_hot(label,depth=3,off_value=0.0)
    #return ([id,matrix],[id,label])
    #return ({'id_with_index':id,'embedding_matrix':matrix},{'id_with_index':id,'dense_softmax':label})
    #return ({'id_with_index':id,'embedding_matrix':matrix},{'dense_softmax':label})
    #return ({'embedding_matrix': matrix}, {'dense_softmax': label})
    return matrix,label



def make_dataset(datapath,train):
    assert os.path.exists(datapath)
    dataset=tf.data.TFRecordDataset(datapath)
    dataset=dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if train:
        dataset=dataset.shuffle(buffer_size=2048)
    dataset=dataset.map(map_func=parse_fn,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if train:
        dataset=dataset.batch(batch_size=128)
    else:
        dataset = dataset.batch(batch_size=256)
    return dataset

def make_model():
    input=keras.Input(shape=(128,4096),name='input_embmat')
    rnn1=keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True),name='biRNN1')(input)
    conv1d1=keras.layers.Conv1D(32,3,activation='relu',name='conv1d')(rnn1)
    rnn2=keras.layers.Bidirectional(keras.layers.LSTM(126),name='biRNN2')(conv1d1)
    reshape1=keras.layers.Reshape((rnn2.shape[1],1),name='reshape1')(rnn2)
    conv1d2=keras.layers.Conv1D(5,5,strides=2,activation='relu',name='conv1d2')(reshape1)
    flatten1=keras.layers.Flatten(name='flatten1')(conv1d2)
    dense1=keras.layers.Dense(32,activation='relu',name='dense1')(flatten1)
    dropout1=keras.layers.Dropout(0.3,name='dropout1')(dense1)
    dense2=keras.layers.Dense(3,activation='softmax',name='dense2')(dropout1)
    model=keras.Model(inputs=input,outputs=dense2)
    model.summary()
    return model
if __name__ == '__main__':
    train_datapath='~/DataSets/1t/df1/extract_features_time_10_26_14_34_size_227404.tf_record'
    val_datapath='~/DataSets/1t/df1/extract_features_time_10_26_17_42_size_59197.tf_record'
    pred_datapath='~/DataSets/1t/df1/extract_features_time_10_26_13_51_size_280339.tf_record'
    train_datapath=os.path.expanduser(train_datapath)
    val_datapath=os.path.expanduser(val_datapath)
    pred_datapath = os.path.expanduser(pred_datapath)
    if TRAIN:
        train_dataset=make_dataset(datapath=train_datapath,train=TRAIN)
        val_dataset=make_dataset(datapath=val_datapath,train=False)
        # predict_dataset=make_dataset()
        if CHECKPOINT is not None:
            model=keras.models.load_model(CHECKPOINT)
        else:
            model=make_model()
        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss={'dense2':'categorical_crossentropy'},
                      metrics=['accuracy'])
        if DEV:
            history=model.fit(val_dataset,epochs=4,batch_size=128)
            '''
            plot_graphs(history, 'accuracy')
            plot_graphs(history, 'loss')
            '''

        else:
            dev_dataset=make_dataset(datapath=val_datapath,train=True)
            #history=model.fit(dev_dataset,epochs=4,batch_size=128,validation_data=val_dataset,validation_steps=30)
            history = model.fit(train_dataset, epochs=1)
            '''
            test_loss, test_acc = model.evaluate(val_dataset)

            print('Test Loss: {}'.format(test_loss))
            print('Test Accuracy: {}'.format(test_acc))
            plot_graphs(history, 'accuracy')
            plot_graphs(history, 'loss')
            '''
            model.save(DEFAULTCHECKPOINT+'%d.h5'%int(time.time()))
    if PREDICT:
        assert CHECKPOINT is not None
        model = keras.models.load_model(CHECKPOINT)
        predict_dataset = make_dataset(datapath=pred_datapath,train=False)
        pre=model.predict(predict_dataset)
        print(len(pre))
        with open(DEFAULTPREDICT+'/%d'%int(time.time()),'wb') as savepre:
            cPickle.dump(pre,savepre)



    