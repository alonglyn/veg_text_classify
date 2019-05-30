# -*-coding:utf-8-*-
from train_cnn import create_model
from data_help import VegDB
import pandas as pd
from tensorflow import keras

vocab_path = 'model/vocab'
vegdb = VegDB(vocab_path)
maxlen = 500
vocab_size = vegdb.vocab_size
df = pd.read_csv('article/sample.csv', sep='\t')

datas = [i[:500] for i in vegdb.get_predict_data_from_dataframe(df[:100], data_field='text')]

train_data = keras.preprocessing.sequence.pad_sequences(datas,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=maxlen)
# Restore the weights
model = create_model(vocab_size)
model.load_weights('model/veg_lstm')
labels = model.predict(train_data)
label = ['正向', '负向', '未知']

def get_label(p):
    if p > 0.9:
        return 1
    elif p < 0.1:
        return 0
    else:
        return 2

with open('example.csv','w') as f:
    for i in range(100):
        f.write('%s\t%s\n' % (' '.join(vegdb.decode_sentence(datas[i])), label[get_label(labels[i])]))
