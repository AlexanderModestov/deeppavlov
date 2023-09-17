import nltk
import re
import pymorphy2
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Widen list of stop-words:
stop_words = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()

# Напишем функцию составления слов:
def get_words(sentence):
    words = nltk.word_tokenize(sentence)
    temp=[]
    for word in words:
        word = re.sub('[^a-zа-я-]', '', word.lower())
        if len(word)<4: continue
        word = morph.parse(word)
        if word[0].normal_form in stop_words: continue
        temp.append(word[0].normal_form)
    return temp