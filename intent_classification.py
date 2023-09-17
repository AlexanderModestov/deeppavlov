# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 00:20:12 2022

@author: Alexa
"""

# Libraries:
import pandas as pd
import re

# Count the quantity of words
from collections import Counter

# лематизация
import pymorphy2

# Test-Train
from sklearn.model_selection import train_test_split # Можно еще KFold рассмотреть для кросс-валидации 

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# расширяем стандартный набор стоп слов
stop_words=set(stopwords.words('russian')

# Upload DataFrame:
df = pd.read_csv('intents.csv', delimiter=';', encoding='cp1251')

#Составим обучающие и тестовые наборы данных:
train, test = train_test_split(df, test_size=0.2) #random_state=42) #размер теста - 30%

# лематизация
morph = pymorphy2.MorphAnalyzer()

# Напишем функцию составления слов:
def get_words(sentence):
    words = nltk.word_tokenize(sentence)
    without_stop_words = [word for word in words if not word in stop_words]
    temp=[]
    for w in without_stop_words:
        t2=re.sub('[^a-zа-я-]', '', w.lower())
        if len(t2)<4: continue # отбрасываю слова короче трех символов
        if t2 in stop_words: continue
        p= morph.parse(t2)
        if p[0].normal_form in stop_words: continue
        temp.append(p[0].normal_form)
    return temp

def concat_lists(list_of_lists):
    result = []
    for list in list_of_lists:
        for item in list:
            result.append(item)
    return result

def top_n(counter):
    parameter = 20
    key_points = counter.most_common(parameter)
    key_words = []
    for item in key_points:
        key_words.append(item[0])
    return key_words

def interception(list_of_words, key_words):
    return set(list_of_words).intersection(key_words)

# Составим список слов:
df.loc[:, 'words'] = df.loc[:, 'content'].apply(get_words)

# Выбор ключевых слов для категории:
res = train.groupby('intent')['words'].apply(list).reset_index()

res.loc[:, 'words'] = res.loc[:, 'words'].apply(concat_lists)
res.loc[:, 'words'] = res.loc[:, 'words'].apply(Counter)
res.loc[:, 'words'] = res.loc[:, 'words'].apply(top_n)
