import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import os, pickle, re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback


stop_eng = stopwords.words('english')

train_path = './data/train.csv.zip'
test_path = './data/test.csv.zip'
embedding_path = './data/crawl-300d-2M.vec'

class Preprocess:
    def __init__(self, maxlen, embed_dim, minCount,train_path=train_path,test_path=test_path):
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.minCount = minCount
        self.df_train = self.load_data(train_path)
        self.df_test = self.load_data(test_path)

    def __call__(self):
        X_train = self.fit(self.df_train['comment_text'],True)
        y_train = self.df_train.iloc[:,2:].values
        test = self.fit(self.df_test['comment_text'],False)
        return X_train, y_train, test

    def fit(self,df,is_train):
        if is_train:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(df)
            self.tokenizer = tokenizer
            self.max_features = max(tokenizer.index_word) + 1

        df=self.tokenizer.texts_to_sequences(df)
        df = pad_sequences(df, maxlen=self.maxlen)
        return df

    def load_data(self,path):
        return pd.read_csv(path, compression='zip')

def load_pre_embedding(preprocessor,embedding_path=embedding_path):
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_path))

    index_word = preprocessor.tokenizer.index_word
    max_features = preprocessor.max_features
    embedding_matrix = np.zeros((max_features, preprocessor.embed_dim))
    for i, word in index_word.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
