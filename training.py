#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import pandas as pd
import re
from keras.utils.np_utils import to_categorical
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.Session(config=tf.ConfigProto())
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4" # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
KTF.set_session(session)


# In[2]:


learning_rate = 5e-5
min_learning_rate = 1e-5
config_path = './uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './uncased_L-12_H-768_A-12/vocab.txt'
MAX_LEN = 512


# In[3]:


token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)


# In[4]:


epochs = 5
save_path = "./model/albert_epoch{0}/".format(epochs)
if not os.path.exists(save_path):    
    os.mkdir(save_path)
    
if not os.path.exists(save_path+"submission/"):    
    os.mkdir(save_path+"submission/")    
    
if not os.path.exists(save_path+"log/"):    
    os.mkdir(save_path+"log/")


# In[5]:


file_path = save_path+"log/"
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)


# In[6]:


df = pd.read_csv('./data/syn_sentence_pair.csv')

df =df.sample(frac = 0.1, random_state=1)
# In[7]:
df1,df2,l1,l2 = train_test_split(df.drop(['label'],axis=1),df['label'],test_size=0.2)

# train_achievements = df['description'].values
# train_requirements = df['abstract'].values
# labels = df['label'].astype(int).values
# test_achievements = df['description'].head(100).values
# test_requirements = df['abstract'].head(100).values
train_achievements = df1['description'].values
train_requirements = df1['abstract'].values
labels = l1.astype(int).values
test_achievements = df2['description'].values
test_requirements = df2['abstract'].values
test_labels = l2.astype(int).values


# In[8]:


labels


# In[9]:


# labels_cat = to_categorical(labels)
# labels_cat = labels_cat.astype(np.int32)
# labels_cat.shape
labels_cat = to_categorical(labels)
labels_cat = labels_cat.astype(np.int32)
labels_cat.shape

labels_cat2 = to_categorical(test_labels)
labels_cat2 = labels_cat2.astype(np.int32)
labels_cat2.shape

# In[22]:


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2, y = self.data
            idxs = list(range(len(self.data[0])))
            np.random.shuffle(idxs)
            T, T_, Y = [], [], []
            for c, i in enumerate(idxs):
                achievements = str(X1[i])[:256]
                requirements = str(X2[i])[:256]
                t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=512)
                T.append(t)
                T_.append(t_)
                Y.append(y[i])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(T)
                    T_ = np.array(T_)
                    Y = np.array(Y)
                    yield [T, T_], Y
                    T, T_, Y = [], [], []


# In[ ]:





# In[23]:


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback


# In[24]:


def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))

    T = bert_model([T1, T2])

    T = Lambda(lambda x: x[:, 0])(T)

    output = Dense(2, activation='softmax')(T)

    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),
        metrics=['accuracy']
    )
    model.summary()
    return model

from keras_albert_model import build_albert
def get_albert_model():
    albert_model = build_albert(token_num=30000, training=False, output_layers=[-1, -2, -3, -4])
    # for layer in albert_model.layers:
    #     layer.trainable = False
    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    T = albert_model([T1, T2])
    T = Lambda(lambda x: x[:, 0])(T)
    output = Dense(2, activation='softmax')(T)
    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),
        metrics=['accuracy']
    )
    model.summary()
    return model

# In[25]:


class Evaluate(Callback):
    def __init__(self, val_data, val_index,model_path):
        self.score = []
        self.best = 0.
        self.early_stopping = 0
        self.val_data = val_data
        self.val_index = val_index
        self.predict = []
        self.lr = 0
        self.passed = 0
        self.model_path = model_path

    def on_batch_begin(self, batch, logs=None):
        if self.passed < self.params['steps']:
            self.lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            self.lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            self.lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, self.lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        score, acc, f1 = self.evaluate()
        if acc > self.best:
            self.best = acc
            self.early_stopping = 0
            model.save_weights(self.model_path)
        else:
            self.early_stopping += 1
        logger.info('lr: %.6f, epoch: %d, score: %.4f, acc: %.4f, f1: %.4f,best: %.4f\n' % (self.lr, epoch, score, acc, f1, self.best))

    def evaluate(self):
        self.predict = []
        prob = []
        val_x1, val_x2, val_y, val_cat = self.val_data
        for i in tqdm(range(len(val_x1))):
            achievements = str(val_x1[i])[:256]
            requirements = str(val_x2[i])[:256]

            t1, t1_ = tokenizer.encode(first=achievements, second=requirements)
            T1, T1_ = np.array([t1]), np.array([t1_])
            _prob = model.predict([T1, T1_])
            oof_train[self.val_index[i]] = _prob[0]
            self.predict.append(np.argmax(_prob, axis=1)[0])
            prob.append(_prob[0])

        score = 1.0 / (1 + mean_absolute_error(val_y, self.predict))
        acc = accuracy_score(val_y, self.predict)
        f1 = f1_score(val_y, self.predict, average='macro')
        return score, acc, f1


# In[26]:


class predict_generator:
    def __init__(self, data, batch_size=256):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            X1, X2 = self.data
            idxs = list(range(len(self.data[0])))
            T, T_, = [], []
            for c, i in enumerate(idxs):
                achievements = str(X1[i])[:256]
                requirements = str(X2[i])[:256]
                t, t_ = tokenizer.encode(first=achievements, second=requirements, max_len=512)
                T.append(t)
                T_.append(t_)
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = np.array(T)
                    T_ = np.array(T_)
                    yield [T, T_]
                    T, T_ = [], []


# In[ ]:





# In[27]:


train_requirements.shape


# In[28]:


train_achievements.shape


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_train = np.zeros((len(train_achievements), 2), dtype=np.float32)
oof_test = np.zeros((len(test_requirements), 2), dtype=np.float32)
for fold, (train_index, valid_index) in enumerate(skf.split(train_achievements, labels)):
    logger.info('================     fold {}        ==============='.format(fold))
    x1 = train_achievements[train_index]
    x2 = train_requirements[train_index]
    y = labels_cat[train_index]

    val_x1 = train_achievements[valid_index]
    val_x2 = train_requirements[valid_index]
    val_y = labels[valid_index]
    val_cat = labels_cat[valid_index]

    train_D = data_generator([x1, x2, y])
    model_save_path = save_path + "BERTModel_{0}.weights".format(str(fold))
    evaluator = Evaluate([val_x1, val_x2, val_y, val_cat], valid_index,model_save_path)

    # model = get_model()
    model = get_albert_model()
    
    model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=epochs,
                        callbacks=[evaluator]
                       )
    model.load_weights(model_save_path)
    
    test_D = predict_generator([test_achievements, test_requirements])
    oof_test += model.predict_generator(test_D.__iter__(), steps=len(test_D))

    #####################
    tprob = np.argmax(oof_test, axis=1)
    print(tprob)
    score_test = 1.0 / (1 + mean_absolute_error(test_labels, tprob))
    acc_test = accuracy_score(test_labels, tprob)
    f1_test = f1_score(test_labels, tprob, average='macro')
    print('score',score_test)
    print('acc',acc_test)
    print('f1',f1_test)
    #####################


    # print(oof_test)
    break
    K.clear_session()
oof_test /= 5





