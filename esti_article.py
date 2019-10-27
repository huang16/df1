import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle as cPickle
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from dataclean import DataExample


TRAIN=False
DEV=False
VAL=True
PRED=True
CHECKPOINT='./checkpoint/atkmodel1572169532.h5'
FILEPATH='~/DataSets/1t/df1/'
ASSEMBLEFOLDER='~/DataSets/1t/df1/'
ANSWERPATH='~/DataSets/df1/Train_DataSet_Label.csv'
PREDICTSEQ='~/DataSets/df1/submit_example.csv'
DEFAULTCHECKPOINT='./checkpoint/atkmodel'
DEFAULTPREDICT='./predict'
FILEPATH=os.path.expanduser(FILEPATH)
ASSEMBLEFOLDER=os.path.expanduser(ASSEMBLEFOLDER)
ANSWERPATH=os.path.expanduser(ANSWERPATH)
PREDICTSEQ=os.path.expanduser(PREDICTSEQ)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()



def load_answers(answerpath):
    answer = pd.read_csv(answerpath, index_col=0)
    ans_dict = answer.to_dict()['label']
    return ans_dict
def assemble(raw_path,vector_path,pred=False):
    with open(FILEPATH+raw_path,'rb') as rawfile:
        rawdata=cPickle.load(rawfile)
    with open(FILEPATH+vector_path,'rb') as vectorfile:
        vectordata=cPickle.load(vectorfile)
    assert len(rawdata)==len(vectordata)
    assemble_dict={}
    maxlen=0
    for i in range(len(rawdata)):
        id=rawdata[i]['id']
        if id[0:32] in assemble_dict.keys():
            listptr=assemble_dict[id[0:32]]
            listptr.append(vectordata[i])
            if len(assemble_dict[id[0:32]])>maxlen:
                maxlen=len(assemble_dict[id[0:32]])
        else:
            assemble_dict[id[0:32]]=[vectordata[i]]
    print(maxlen)
    assemble_vector=[]
    assemble_label=[]
    if not pred:
        ans_dict=load_answers(ANSWERPATH)
        for key in assemble_dict.keys():
            assemble_vector.append(assemble_dict[key])
            assemble_label.append(ans_dict[key])
        with open(ASSEMBLEFOLDER+'%d_%d'%(len(assemble_vector),int(time.time()))+'assemble_vextor','wb') as vectorptr:
            cPickle.dump(assemble_vector,vectorptr)
        with open(ASSEMBLEFOLDER+'%d_%d'%(len(assemble_label),int(time.time()))+'assemble_label','wb') as labelptr:
            cPickle.dump(assemble_label,labelptr)
        return assemble_vector,assemble_label
    else:
        ans_dict = load_answers(PREDICTSEQ)
        for key in ans_dict:
            assemble_vector.append(assemble_dict[key])
        with open(ASSEMBLEFOLDER + 'pred%d_%d' % (len(assemble_vector), int(time.time())) + 'assemble_vextor',
                      'wb') as vectorptr:
            cPickle.dump(assemble_vector, vectorptr)
        return assemble_vector

def make_model():
    input=keras.Input(shape=(256,3),name='input_embmat')
    rnn1=keras.layers.Bidirectional(keras.layers.LSTM(128,return_sequences=True),name='biRNN1')(input)
    conv1d1=keras.layers.Conv1D(32,3,activation='relu',name='conv1d')(rnn1)
    rnn2=keras.layers.Bidirectional(keras.layers.LSTM(64),name='biRNN2')(conv1d1)
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


    # TODO modify to runnable
    train_filename=['OUTPUT_Train_DataSet.csv227404','1572160301']
    val_filename=['OUTPUT_Train_DataSet.csv59197','1572160935']
    pred_filename=['OUTPUT_Test_DataSet.csv280339','1572151553']
    if TRAIN:
        train_ax,train_ay= assemble(train_filename[0],train_filename[1],pred=False)
        train_x=keras.preprocessing.sequence.pad_sequences(train_ax,value=[0,0,0],padding='post',maxlen=256)
        train_y=np.eye(3)[train_ay]
        train_dataset=tf.data.Dataset.from_tensor_slices((train_x,train_y))
        train_dataset.batch(128)
        if CHECKPOINT is not None:
            model=keras.models.load_model(CHECKPOINT)
        else:
            model=make_model()
        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        if DEV:
            history=model.fit(train_x,train_y,epochs=16,batch_size=128)
            '''
            plot_graphs(history, 'accuracy')
            plot_graphs(history, 'loss')
            '''
        else:
            #history = model.fit(train_x,train_y, epochs=32, batch_size=128, validation_data=val_dataset, validation_steps=30)
            history = model.fit(train_x,train_y, epochs=16, batch_size=128)
            model.save(DEFAULTCHECKPOINT + '%d.h5' % int(time.time()))
            val_ax,val_ay=assemble(val_filename[0],val_filename[1],pred=False)
            #val_dataset=tf.data.Dataset.from_tensor_slices((np.array(val_x),np.array(val_y)))
            val_x = keras.preprocessing.sequence.pad_sequences(val_ax, value=[0, 0, 0], padding='post', maxlen=256)
            val_y = np.eye(3)[val_ay]
            test_loss, test_acc = model.evaluate(val_x,val_y,batch_size=128)

            print('Test Loss: {}'.format(test_loss))
            print('Test Accuracy: {}'.format(test_acc))
            '''
            plot_graphs(history, 'accuracy')
            plot_graphs(history, 'loss')
            '''


    if PRED:
        assert CHECKPOINT is not None
        model = keras.models.load_model(CHECKPOINT)
        predict_ax= assemble(pred_filename[0],pred_filename[1],pred=True)
        predict_x = keras.preprocessing.sequence.pad_sequences(predict_ax, value=[0, 0, 0], padding='post', maxlen=256)
        pred_dataset=tf.data.Dataset.from_tensor_slices((predict_x))
        #pred_dataset.batch(256)
        pre=model.predict(predict_x)
        print(len(pre))
        with open(DEFAULTPREDICT+'/article%d'%int(time.time()),'wb') as savepre:
            cPickle.dump(pre,savepre)



