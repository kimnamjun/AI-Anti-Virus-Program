def predict_one(df, model):
    x_test = df
    y_pred = model.predict(x_test)
    return y_pred


def predict_two(df, model):
    x_test = df['imports'].apply(lambda row: ' '.join(row)).to_list()
    y_pred = model.predict(x_test)
    return y_pred


""" 모델 참고
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, Dropout, Embedding, LSTM

class MyModel(Model):
    def __init__(self, train_dataset):
        super(MyModel, self).__init__()
        self.dataset = train_dataset
        self.text_vectorization = TextVectorization(output_mode='int', output_sequence_length=400)
        self.text_vectorization.adapt(self.dataset.map(lambda text, label: text))
        self.embedding = Embedding(len(self.text_vectorization.get_vocabulary()), 128, mask_zero=True)
        self.bidirectional1 = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))
        self.bidirectional2 = Bidirectional (LSTM(64, dropout=0.5, return_sequences=True, return_state=True))
        self.concatenate_c = Concatenate()
        self. concatenate_h = Concatenate()
        self.w1 = Dense(64)
        self.w2 = Dense(64)
        self.v = Dense(1)
        self.dense1 = Dense(20, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.text_vectorization(x)
        x = self.embedding(x)
        x = self.bidirectional1(x)
        x, fh, fc, bh, bc = self.bidirectional2(x)
        h = self.concatenate_h((fh, bh))
        c = self.concatenate_c((fc, bc))
        x = tf.nn.softmax(self.v(tf.nn.tanh(self.w1(x) + self.w2(tf.expand_dims(h, 1)))), axis=1) * x
        x = tf.reduce_sum(x, axis=1)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
"""
