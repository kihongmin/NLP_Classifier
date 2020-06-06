from collections import Counter
from nltk.corpus import stopwords
import os, pickle, re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
stop_eng = stopwords.words('english')

class Preprocess:
    def __init__(self, maxlen, embed_dim, minCount):
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.minCount = minCount

    def __call__(self,train,test):
        train = self.fit(train,True)
        test = self.fit(test,False)
        return train,test

    def fit(self,df,is_train):
        if is_train:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(df)
            self.tokenizer = tokenizer

        df=self.tokenizer.texts_to_sequences(df)
        df = pad_sequences(df, maxlen=self.maxlen)
        return df
