import pandas as pd
import pymorphy2
#import deeppavlov
import data_preprocessing

df = pd.read_csv('intents.csv', delimiter=';', encoding='cp1251')

# Lemmatization:
morph = pymorphy2.MorphAnalyzer()

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Stop - words:
stop_words = stopwords.words('russian')
print(stop_words)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')