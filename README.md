2019-5-29花了一个晚上重新实现了一遍以前的任务

# 任务目的

主要目的是使用神经网络预测文本的情感倾向，这里是预测蔬菜报道针对蔬菜价格走势的判断。

认为蔬菜价格走高， 则标记为正向文本， 反之则是负向。

使用的神经网络模型是最基本的bi-lstm。

主要任务是预处理文本

# 最终结果

可以对一系列相关蔬菜价格文本做关于价格走势的预测

# 需要的工具
分词工具:jieba

word2vec:gensim

深度学习框架:keras

其他：python3的基础工具包

# 参考资料搜索关键词
keras 文本情感分类
深度学习情感分析



文本主题分类
文本多分类任务、神经网络



# 目录树以及关键的数据

├── article

│   ├── all.xlsx

│   └── sample.csv	//待预测文章采样

├── corpus

│   ├── corpus.csv													//语料库、有标注训练集

│   ├── stopwords.txt											//停用词

│   ├── word2vec													//word2vec的语料库， 可以将all.xlsx的数据一并拿来训练

│   ├── 负向														//抽取的正向文本

│   └── 正向														//抽取的负向文本

├── data_help.py

├── example.csv

├── model

│   ├── checkpoint

│   ├── checkpoints.data-00000-of-00001

│   ├── checkpoints.index

│   ├── veg_lstm.data-00000-of-00001				//预测模型

│   ├── veg_lstm.index

│   └── vocab															//词表、第一次使用vegdb的时候需要提供corpus构建。

├── predict.py														//加载模型进行预测， 示例结果存放在example.csv里

├── README.md

├── train_lstm.py													//lstm训练模型， 训练数据路径、参数在代码里面改

└── train_wordvec_model.py                               //预训练词向量（这里没有使用）




## 主要的代码

- data_help.py

  提供vegdb类， 可以建立一个语料库以及相应的词表

  提供接口从dataframe表中抽取训练数据、预测数据

  也就是将中文文本转换成int数组

- train_word2vec.py

  可以得到word2vec预训练的词向量

- train_lstm.py

  - 切分训练集、测试集
  - 句子进行长度为500的零填充
  - 构建lstm模型， 使用二分类输出层。其他参数没有进行调节。

- predict.py

  - 提供预测功能
  - 预测的数据标签二元组， 保存再example.csv里


