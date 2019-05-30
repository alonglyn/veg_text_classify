# -*-coding:utf-8-*-
from data_help import VegDB
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_dataset(data, shuffle=True, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data[1]))
    ds = ds.batch(batch_size)
    return ds


def create_model(vocab_size):
    '''构建网络模型
    嵌入层64维
    64个双向LSTM单元
    64个全连接单元
    sigmoid输出层
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    '''编译运行
    使用了二项交叉熵、其实就是逻辑回归损失函数
    '''
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    '''从corpus里抽取训练数据'''
    from_path = 'corpus/corpus.csv'
    data_field = '文本'
    cate_field = '情感'
    to_dir = 'corpus'
    df = pd.read_csv(from_path, sep='\t')
    vocab_path = 'model/vocab'

    if vocab_path == '':
        vegdb = VegDB()
        vegdb.create_vocab(df, data_field='文本')
    else:
        vegdb = VegDB(vocab_path)

    '''从vegdb中抽取训练数据以及参数'''

    (train_data, train_label), (test_data, test_label) = vegdb.get_train_data_from_dataframe(df=df, data_field='文本',
                                                                                         cate_field='情感')
    maxlen = 500 #max(max([len(i) for i in test_data]), max([len(i) for i in train_data]))
    vocab_size = vegdb.vocab_size
    print(vocab_size, maxlen)
    # padding with maxlen
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=maxlen)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=0,
                                                           padding='post',
                                                           maxlen=maxlen)
    # pack up to dataset
    train_ds = get_dataset((train_data, train_label))
    test_ds = get_dataset((test_data, test_label))

    '''
    训练并保存模型
    '''
    checkpoint_path = "model/checkpoints"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model = create_model(vocab_size)

    history = model.fit(train_ds, epochs=10,
                        validation_data=test_ds,
                        callbacks=[cp_callback])

    # Save the weights
    model.save_weights('model/veg_lstm')
