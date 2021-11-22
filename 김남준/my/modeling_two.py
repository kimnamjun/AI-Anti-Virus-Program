import numpy as np
import pandas as pd
from gensim.models import Word2Vec

def word_to_vector(train_df: pd.DataFrame, test_df: pd.DataFrame, **word2vector_config) -> tuple:
    x_train, y_train = train_df['imports'], train_df['label']
    x_test, y_test = test_df['imports'], test_df['label']

    word2vector = Word2Vec(x_train, word2vector_config)
    word2vector.init_sims(replace=True)
    word_vector_keys = word2vector.wv.vocab.keys()

    x_train_vector, y_train_vector = list(), list()
    for idx in range(len(x_train)):
        inputs = list()
        for word in x_train[idx]:
            if word in word_vector_keys:
                inputs.append(word2vector.wv[word].tolist())
            else:
                break
        else:
            x_train_vector.append(inputs)
            y_train_vector.append(y_train[idx])

    x_test_vector, y_test_vector = list(), list()
    for idx in range(len(x_test)):
        inputs = list()
        for word in x_test[idx]:
            if word in word_vector_keys:
                inputs.append(word2vector.wv[word].tolist())
            else:
                break
        else:
            x_test_vector.append(inputs)
            y_test_vector.append(y_test[idx])

    x_train_vector = np.array(x_train_vector, dtype='float32')
    y_train_vector = np.array(y_train_vector, dtype='float32')

    x_test_vector = np.array(x_test_vector, dtype='float32')
    y_test_vector = np.array(y_test_vector, dtype='float32')

    return x_train_vector, y_train_vector, x_test_vector, y_test_vector

def run_lstm():
    pass