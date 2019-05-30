# -*-coding:utf-8-*-
import jieba
import os
import pandas as pd
import re
import numpy as np

with open('corpus/stopwords.txt') as f:
    stopwords = f.read().split('\n')


def cut(s):
    return [i for i in jieba.cut(s) if is_valid(i)]


def is_valid(s):
    return s not in stopwords and re.match('[\d]+', s) == None


class VegDB:
    def __init__(self, vocab_path=''):
        if vocab_path != '':
            with open(vocab_path, 'r') as fin:
                self.vocab = fin.read().split(' ')
                self.generate_index()

    def generate_index(self):
        self.vocab_size = len(self.vocab)
        self.word_index = dict((v, i) for i, v in enumerate(self.vocab))

    def create_vocab(self, df, data_field):
        datas = [cut(s) for s in list(df[data_field]) if s]
        self.vocab = ['<PAD>', '<UNK>']
        for i in datas:
            self.vocab.extend(i)
        self.vocab = list(set(self.vocab))
        self.generate_index()
        with open('model/vocab', 'w') as fout:
            fout.write(' '.join(self.vocab))

    '''获得训练数据
    输入是一个dataframe等
    输出是训练集、数据集、还有附加信息
    '''

    def get_train_data_from_dataframe(self, df, data_field, cate_field, shuffle=True, test_rate=0.1):
        df = df[(df[cate_field] == '正向') | (df[cate_field] == '负向')]
        datas = [cut(s) for s in list(df[data_field]) if s]
        labels = list(df[cate_field])
        # info = dict()
        # info['maxlen'] = max([len(i) for i in datas])
        label_index = {'正向': 0, '负向': 1}  # dict((v, i) for i, v in enumerate(set(labels)))
        datas = [self.encode_sentence(s) for s in datas]
        labels = [label_index[s] for s in labels]
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(datas)
            np.random.set_state(state)
            np.random.shuffle(labels)
        boundary = int(len(labels) * (1 - test_rate))
        return (datas[:boundary], labels[:boundary]), (datas[boundary:], labels[boundary:])

    '''编码预测数据
    输入是dataframe
    输出是编码后的数据
    '''

    def get_predict_data_from_dataframe(self, df, data_field):
        datas = [self.encode_sentence(cut(s)) for s in list(df[data_field]) if s]
        return datas

    def get_word_index(self):
        return self.word_index

    def encode_sentence(self, s):
        # if i in self.word_index.keys() else 1
        return [self.word_index[i] if i in self.word_index.keys() else 1 for i in s]

    def decode_sentence(self, s):
        return [self.vocab[i] for i in s]


'''

def extract_from_corpus(from_path, data_field, cate_field, to_dir):
    df = pd.read_csv(from_path, sep='\t')
    cates = ['正向', '负向']
    for cate in cates:
        with open(os.path.join(to_dir, cate), 'w') as f:
            f.write('\n'.join([' '.join(cut(i)) for i in df[df[cate_field] == cate][data_field]]))
    return cates


'''
