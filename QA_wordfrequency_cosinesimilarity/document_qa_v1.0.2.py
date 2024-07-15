# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:38:27 2024

@author: fengzenghang
"""

import re
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

##############################################################################
# 知识库数据处理
##############################################################################
# 1. 读取文档数据，进行数据预处理 ===============================================
# 1.1 读取停用词词库 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_stop_words(stopword_txt):
    """
    Parameters
    ----------
    stopword_txt : TYPE 字符串
        DESCRIPTION. 存储停用词的文本文档路径

    Returns
    -------
    stop_w_l : TYPE 列表
        DESCRIPTION. 返回存储停用词的列表
    """
    with open(stopword_txt, 'r', encoding='utf-8') as f:
        stop_w_l = [word.strip('\n') for word in f.readlines()]
    return stop_w_l
    

def get_key_words(knowledge_docs, stop_w_l):
    """
    Parameters
    ----------
    knowledge_docs : TYPE 字符串
        DESCRIPTION. 存储知识库各个文档的文档路径
    stop_w_l : TYPE 列表
        DESCRIPTION. 存储停用词的列表

    Returns
    -------
    df_klg : TYPE Dataframe
        DESCRIPTION. 存储知识库各个文档的Dataframe
    doc_keywords : TYPE 列表
        DESCRIPTION. 存储知识库各个文档的关键词列表
    """
    
    # 1.2 读取文档库，并进行分词和去除停用词处理 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_klg = pd.read_excel('knowledge_docs.xlsx')
    # 对文档分词，生成的结果存储在列表中，每个词作为列表中一个元素
    df_klg['文档内容_分词'] = df_klg.apply(lambda row: jieba.lcut(re.sub(r'[\W]', '', row['文档内容'])), axis=1)
    # 去除停用词，生成的结果存储在字符串中，每个词之间以空格分隔
    df_klg['文档内容_分词_去停用词'] = df_klg.apply(lambda row: ' '.join([w for w in row['文档内容_分词'] if w not in stop_w_l]), axis=1)
    
    # 2. 计算文档数据的TF-IDF，提取关键词 ===========================================
    # 2.1 词矢量化矩阵 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    klg_text = df_klg['文档内容_分词_去停用词'].tolist()
    vectorizer = CountVectorizer()
    word_vect = vectorizer.fit_transform(klg_text)
    word_list = vectorizer.get_feature_names()
    
    # 2.2 计算TF-IDF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    transform = TfidfTransformer()
    tf_idf = transform.fit_transform(word_vect.toarray())
    tf_idf_ary = tf_idf.toarray()
    
    # 2.3 根据TF-IDF数值提取问答关键词 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    num_filt = 10                               # 如果词的数量大于10，保留10个关键词
    doc_keywords = []                           # 存放知识库中每个文档提取的关键词
    for row in tf_idf_ary:
        val_filt = np.sum(row > 0)              # 每一个文档中大于0的元素个数
        
        if val_filt >= num_filt:
            list_key_words = row.argsort()[-num_filt:].tolist()
        else:
            list_key_words = row.argsort()[-val_filt:].tolist()
        
        doc_keywords.append(' '.join([word_list[t] for t in list_key_words]))
        
    return df_klg, doc_keywords


##############################################################################
# 问题数据处理
##############################################################################
def get_question(question_txt, stop_w_l):
    """
    Parameters
    ----------
    question_txt : TYPE 字符串
        DESCRIPTION. 用户输入的问题的字符串
    stop_w_l : TYPE 列表
        DESCRIPTION. 存储停用词的列表

    Returns
    -------
    question_spl : TYPE 字符串
        DESCRIPTION. 用户输入的问题的分词结果

    """
    question_txt = re.sub(r'[\W]', '', question_txt)
    question_spl = jieba.lcut(question_txt)
    question_spl = ' '.join([w for w in question_spl if w not in stop_w_l])
    
    return question_spl


##############################################################################
# 计算问题关键词向量与知识库各个文档关键词向量的余弦相似度，得到答案
##############################################################################
def get_answer(question_spl, df_klg, doc_keywords):
    """
    Parameters
    ----------
    question_spl : TYPE 字符串
        DESCRIPTION. 用户输入的问题的分词结果
    df_klg : TYPE Dataframe
        DESCRIPTION. 存储知识库各个文档的Dataframe
    doc_keywords : TYPE 列表
        DESCRIPTION. 存储知识库各个文档的关键词列表

    Returns
    -------
    answer_doc : TYPE 字符串
        DESCRIPTION. 针对用户查询问题返回的文档内容

    """
    input_str = question_spl
    sim_res = []
    for key_words in doc_keywords:
        vectorizer = CountVectorizer()
        word_vect_tmp = vectorizer.fit_transform([key_words, input_str]).toarray()
        cos_s = cosine_similarity(word_vect_tmp[0].reshape(1,-1), word_vect_tmp[1].reshape(1,-1))
        sim_res.append(cos_s[0,0])
    
    sim_res_ary = np.array(sim_res)
    index_sort = np.argsort(sim_res_ary)        # 返回按照余弦相似度数值从小到大排序的索引值
    # cos_val_sort = np.sort(sim_res_ary)         # 返回按照余弦相似度数值从小到大排序的数值
    
    index_max = index_sort[-1]                  # 得到余弦相似度最大数值对应的索引
    answer_doc = df_klg.loc[index_max, '文档内容']       # 根据索引从知识库中查询答案并返回
    
    return answer_doc


if __name__ == '__main__':
    question_txt = input('请输入需要查询的问题：')
    # 示例：'中国和美国哪个国家的经济更好一些？'
    
    stopword_txt = 'hit_stopwords.txt'
    stop_w_l = get_stop_words(stopword_txt=stopword_txt)
    
    knowledge_docs = 'knowledge_docs.xlsx'
    df_klg, doc_keywords = get_key_words(knowledge_docs, stop_w_l)
    
    question_spl = get_question(question_txt, stop_w_l)
    
    answer_doc = get_answer(question_spl, df_klg, doc_keywords)