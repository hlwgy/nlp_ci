# coding=utf-8
#%% 导入包
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# %% 加载数据
def load_data(num = 1000):
      
    # 读取csv文件。表头：0 题目| 1 作者| 2 内容
    csv_reader = csv.reader(open("./ci.csv",encoding="gbk"))
    # 以一首词为单位存储
    ci_list = []
    for row in csv_reader:
        # 取每一行，找到词内容那一列
        ci_list.append(row[2])
        #　超过最大数量退出循环，用多少取多少
        if len(ci_list) > num:break 
    return ci_list

# 加载数据
def get_train_data():
      
    # 加载数据作为语料库["春花 秋月","一江 春水 向东 流"]
    corpus = load_data()
    # 定义分词器
    tokenizer = Tokenizer()
    # 分词器适配文本，将语料库拆分词语并加索引{"春花":1,"秋月":2,"一江":3}
    tokenizer.fit_on_texts(corpus)

    # 定义输入序列
    input_sequences = []
    # 从语料库取出每一条
    for line in corpus:
        # 序列成数字 "一江 春水 向东 流" ->[3,4,5,6]
        token_list = tokenizer.texts_to_sequences([line])[0]
        # 截取字符[3,4,5,6]变成[3,4],[3,4,5],[3,4,5,6]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    # 找到语料库中最大长度的一项
    max_sequence_len = max([len(x) for x in input_sequences])
    # 填充序列每一项到最大长度，采用前面补0的方式
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    return tokenizer, input_sequences, max_sequence_len

# 构建模型
def create_model(vocab_size, embedding_dim, max_length):
    
    # 构建序列模型
    model = Sequential()
    # 添加嵌入层
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
    # 添加长短时记忆层
    model.add(layers.Bidirectional(layers.LSTM(512)))
    # 添加softmax分类
    model.add(layers.Dense(vocab_size, activation='softmax'))
    # adam优化器
    adam = Adam(lr=0.01)
    # 配置训练参数
    model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])

    return model

# %% 训练数据
def train():
    # 分词器，输入序列，最大序列长度
    tokenizer, input_sequences, max_sequence_len = get_train_data()
    # 得出有多少个词，然后+1，1是统一长度用的填充词
    total_words = len(tokenizer.word_index) + 1
    '''
    至此，我们得到了如下序列
    [0,0,1,2],[0,0,3,4],[0,3,4,5],[3,4,5,6]
    对应文字就是：[0,0,春花,秋月],[0,0,一江,春水],[0,一江,春水,向东],[一江,春水,向东,流]
    '''
    # 从语料库序列中拆分出输入和输出，输入是前面几个词，输出是最后一个词
    xs = input_sequences[:,:-1]
    labels = input_sequences[:,-1]

    # 结果转为独热编码
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    # 创建模型
    model = create_model(total_words, 256, max_sequence_len-1)
    # 进行训练
    model.fit(xs, ys, epochs= 15, verbose=1)

    # 保存训练的模型
    model_json = model.to_json()
    with open('./save/model.json', 'w') as file:
        file.write(model_json)
    # 保存训练的权重
    model.save_weights('./save/model.h5')

# %% 预测数据 给定一个开头词语，给定后面要预测多少个词
def predict(seed_text, next_words = 20):

    # 分词器，输入序列，最大序列长度
    tokenizer, input_sequences, max_sequence_len = get_train_data()

    # 读取训练的模型结果
    with open('./save/model.json', 'r') as file:
        model_json_from = file.read()
    model = tf.keras.models.model_from_json(model_json_from)
    model.load_weights('./save/model.h5')

    # 假如要预测后面next_words=20个词，那么需要循环20词，每次预测一个
    for _ in range(next_words):
        # 将这个词序列化 如传来“高楼”,则从词库中找高楼的索引为50，序列成[50]
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # 填充序列每一项到最大长度，采用前面补0的方式[0,0……50]
        token_list = pad_sequences([token_list], maxlen= max_sequence_len-1, padding='pre')
        # 预测下一个词，预测出来的是索引
        predicted = model.predict_classes(token_list, verbose = 0)
        # 定义一个输出存储输出的数值
        output_word = ''
        # 找到预测的索引是哪一个词，比如55是“灯火”
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        # 输入+输出，作为下一次预测：高楼 灯火
        seed_text = seed_text + " " + output_word

    print(seed_text)
    # 替换空格返回
    return seed_text.replace(' ' , '')

# %%
if __name__ == '__main__':
    # 训练数据
    #train()
    
    # 预测数据
    print(predict('细雨',next_words = 22))
    # 细雨仙桂春。明月此，梦断在愁何。等闲帘寒，归。正在栖鸦啼来。
    #print(predict('清风',next_words = 20))
    # 风到破向，貌成眠无风。人在梦断杜鹃风韵。门外插人莫造。怯霜晨。
    #print(predict('高楼',next_words = 20))
    # 高楼灯火，九街风月。今夜楼外步辇，行时笺散学空。但洗。俯为人间五色。
    #print(predict('海 风',next_words = 20))
    # 海风落今夜，何处凤楼偏好。奇妙。残月破。将心青山上，落分离。
    # print(predict('今夜',next_words = 20))
    # 今夜谁和泪倚阑干。薰风却足轻。似泠愁绪。似清波似玉人。羞见。
