#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:25:19 2021

@author: mm
"""
import csv
import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def txt_to_csv(txt_path, csv_path):#将txt文件转为csv文件
    '''Argument txt_path should be the path of file holder'''
    data_csv = open(csv_path, 'w+', encoding = 'utf-8')
    writer = csv.writer(data_csv)
    writer.writerow(['Content'])
    try:
        dirs = os.listdir(txt_path)
        for file in dirs:
            file_path = txt_path + file
            with open(file_path, 'r', encoding='utf-8', errors = 'ignore') as file_handle:
                content = file_handle.read()
            writer.writerow([content])
    finally:
        data_csv.close()
    return

def test_shape():#测试是否所有200个txt文件都已写入csv文件中
    print('Shape of data =', data.shape)

def cut_sentence(text):#分词
    return ' '.join(jieba.cut(text))

def stwlist(stop_words_path):#导入停用词列表
    stop_words_handle = open(stop_words_path, 'rb')
    stop_words = stop_words_handle.read().decode('utf-8')
    stop_words_list = stop_words.splitlines()
    stop_words_handle.close()
    return stop_words_list
    
def get_top_words(model, feature_names, n_top_words, tw_path):#提取各个主题下关键词
    tw_handle = open(tw_path, 'w')
    for topic_idx, topic in enumerate(model.components_):
        title_line = '主题#%d:' % topic_idx
        tw_handle.write(title_line + ' ')
        words = ' '.join([feature_names[i]                        
    for i in topic.argsort()[:-n_top_words - 1:-1]])
        tw_handle.write(words + '\n')
    tw_handle.close()
    return    

txt_path = r'./files/'
csv_path = r'./data.csv'
stop_words_path = r'./stopwords.txt'
tw_path = r'./top_words.txt'

txt_to_csv(txt_path, csv_path)
data = pd.read_csv(csv_path, encoding = 'utf-8').astype(str)#解决attribute error

test_shape()

data['cutted'] = data.Content.apply(cut_sentence)
stw = stwlist(stop_words_path)

#向量化
n_features = 500
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features = n_features,
                                stop_words = stw,
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(data.cutted)

#LDA
n_components = 5#主题数
lda = LatentDirichletAllocation(n_components = n_components, 
                                max_iter = 50, 
                                learning_method = 'online', 
                                learning_offset = 50., 
                                random_state = 0)
lda.fit(tf)

n_top_words = 30#每个主题下输出30个关键词
tf_feature_names = tf_vectorizer.get_feature_names()
get_top_words(lda, tf_feature_names, n_top_words, tw_path)


 