import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import time
import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def GetTime():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
GetTime()
if __name__ == '__main__':

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # inp为输入语料, outp1 为输出模型, outp2为原始c版本word2vec的vector格式的模型
    # inp为输入语料
    inp = 'corpus/word2vet_txt'
    # outp1 为输出模型
    outp1 = 'model/2019_vec.model'
    # outp2为原始c版本word2vec的vector格式的模型
    outp2 = 'model/2019_vec.vector'

    # 训练skip-gram模型
    model = Word2Vec(LineSentence(inp), size=128, window=5, min_count=3,
                     workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
    print(len(model.wv.vocab))
GetTime()
