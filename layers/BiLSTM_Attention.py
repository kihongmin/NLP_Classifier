import tensorflow as tf
import os, pickle, re
from collections import Counter
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Layer, Bidirectional,Embedding, Dropout, LSTM
from tensorflow.keras import Model, Input
from sklearn.model_selection import train_test_split

tf.random.set_seed(1025)

class Attention(Layer):
    def __init__(self,hidden_size,attention_size=240, **kwargs):
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        self.v_t = Dense(hidden_size,name='v_t',activation='tanh')
        self.w = tf.Variable(tf.random_normal_initializer(stddev=0.1,seed=1025)(shape=[attention_size]),True)
        self.score = Activation('softmax',name='score')

        super(Attention, self).__init__(**kwargs)

    def call(self,hidden_states):
        v = self.v_t(hidden_states)
        vu = tf.tensordot(v, self.w, axes=1, name='vu')

        score = self.score(vu)
        output = tf.reduce_sum(hidden_states * tf.expand_dims(score,-1),1, name='attention_output')

        return output


def build_model(preprocessor,embedding_matrix):
    inp = Input(shape=(preprocessor.maxlen,))
    x = Embedding(preprocessor.max_features,preprocessor.embed_dim,weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(120,return_sequences=True))(x)
    x = Attention(x.shape[2])(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model
