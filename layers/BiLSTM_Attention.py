import tensorflow as tf
import os, pickle, re
from collections import Counter

from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Layer, Bidirectional
from tensorflow.keras import Model

from sklearn.model_selection import train_test_split

stop_eng = stopwords.words('english')

class Attention(Layer):
    def __init__(self,hidden_size, **kwargs):
        self.hidden_size = hidden_size
        self.first_score = Dense(self.hidden_size,name='attention_score_vec')
        self.h_t = Lambda(lambda x:x[:,-1,:], output_shape=(self.hidden_size,),name='last_hidden_state')
        self.softmax = Activation('softmax',name='softmax')
        self.attention_vector = Dense(256, activation='tanh',name='attention_vector')
        super(Attention, self).__init__(**kwargs)

    def call(self,hidden_states):
        first_score = self.first_score(hidden_states)
        h_t = self.h_t(first_score)
        score = dot([first_score,h_t],[2,1],name='attention_score')
        attention_weights = self.softmax(score)
        context_vector = dot([hidden_states,attention_weights],[1,1],name='context_vector')
        pre_activation = concatenate([context_vector,h_t],name='attention_output')
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

def build_model(maxlen=maxlen,embed_dim=embed_dim):
    max_feature = max(preprocessor.tokenizer.index_word.keys())
    inp = Input(shape=(maxlen,))
    x = Embedding(max_feature,embed_dim)(inp)
    x = Bidirectional(LSTM(100,return_sequences=True))(x)
    x = Attention(x.shape[2])(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model
