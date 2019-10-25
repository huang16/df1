import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
train_dataset_path='extract_features_time_10_20_10_48_size_280339.tf_record'
assert os.path.exists(train_dataset_path)
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
    '''
    example = tf.train.Example()
    example.ParseFromString(example.take(1))
    label = example.features.feature['label'].float_list.value[0]
    id = example.features.feature['id'].bytes_list.value[0]
    matrix = example.features.feature['matrix'].bytes_list.value[0]
    #features=tf.parse_single_example(example,example_fmt)
    '''
    print(id)
    print(label)
    print('****************')
    print(repr(matrix))
    matrix=tf.io.decode_raw(matrix,tf.float32)
    #matrix=np.frombuffer(matrix,dtype=np.float32).reshape(128,768,4).transpose((1,2,0))

    #matrix = np.frombuffer(matrix, dtype=np.float32).reshape((4, 128, 768))
    matrix = tf.reshape(matrix, (4, 128, 768))
    matrix = tf.transpose(matrix, perm=[1, 2, 0])
    #label=features['label']
    label=tf.cast(label,dtype=tf.int32)
    label=tf.one_hot(label,depth=3,off_value=0.0)
    #label=tf.reshape(label,(3,1))
    print(matrix)
    print(label)

    #label=features['label'].
    # matrix=np.frombuffer(features['matrix'],dtype=np.float32).reshape(4,128,768)


    #return features['id'],matrix,label
    #return ([id,matrix],[id,label])
    #return ({'id_with_index':id,'embedding_matrix':matrix},{'id_with_index':id,'dense_softmax':label})
    #return ({'id_with_index':id,'embedding_matrix':matrix},{'dense_softmax':label})
    #return ({'embedding_matrix': matrix}, {'dense_softmax': label})
    return matrix,label



def make_dataset():
    dataset=tf.data.TFRecordDataset(train_dataset_path)
    dataset=dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset=dataset.shuffle(buffer_size=1024)
    dataset=dataset.map(map_func=parse_fn,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset=dataset.batch(batch_size=64)
    return dataset


if __name__ == '__main__':
    dataset=make_dataset()
    '''
    matrix = np.random.random_sample(size=(100, 128, 768, 4))
    label = np.random.random_sample(size=(100, 1,3))
    dataset=tf.data.Dataset.from_tensor_slices(({'embedding_matrix':matrix},{'dense_softmax':label}))
    dataset=dataset.batch(8)
    '''
    #id_with_index=keras.Input(shape=(None,),name='id_with_index')
    embedding_matrix=keras.Input(shape=(128,768,4),name='embedding_matrix')
    #conv1=keras.layers.Conv2D(32,(3,3),activation='relu',name='conv1')(embedding_matrix)
    #dense1=keras.layers.Dense(10,activation='relu',name='dense1')(conv1)
    #flatten1=keras.layers.Flatten(name='flatten1')(dense1)
    flatten1 = keras.layers.Flatten(name='flatten1')(embedding_matrix)
    dense2=keras.layers.Dense(3,activation='softmax',name='dense_softmax')(flatten1)
    model=keras.Model(inputs=embedding_matrix,
                    outputs=dense2)
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss={'dense_softmax':'categorical_crossentropy',
                        },
                  metrics=['accuracy'],)
                  #target_tensor=dense2)

    model.summary()
    #keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
    model.fit(dataset,epochs=2)
    pre=model.predict(dataset)
    print(pre)
    #print(pre.shape)

    