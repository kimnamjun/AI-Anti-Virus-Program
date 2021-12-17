import numpy as np
import tensorflow as tf
from datetime import datetime
from pycaret.classification import *

from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, Dropout, Embedding, LSTM


def create_voting_model(train_df, test_df):
    train_df = train_df[train_df['label'] != -1].set_index('sha256')
    test_df = test_df[test_df['label'] != -1].set_index('sha256')

    setup(data=train_df.sample(frac=0.1), target='label', silent=True)
    selected_models = compare_models(n_select=3)
    print('selected models', selected_models, sep='\n')

    params = list()
    for model in selected_models:
        parameters = dict()
        for key, value in model.get_params().items():
            if value is None:
                continue
            parameters[key] = [value]
        params.append(parameters)

    cv = StratifiedKFold(shuffle=True)
    models = [GridSearchCV(estimator=selected_models[i], param_grid=params[i], cv=cv) for i in range(len(selected_models))]

    voting_model = VotingClassifier(estimators=[[str(model.estimator).split('(')[0], model] for model in models])

    x_train, y_train = train_df.drop('label', axis=1), train_df['label']
    x_test, y_test = test_df.drop('label', axis=1), test_df['label']

    voting_model.fit(x_train, y_train)

    y_pred = voting_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('voting model accuracy :', acc)

    return voting_model


def create_tfidf_and_logistic_regression_model(train_df, test_df):
    train_df['corpus'] = [' '.join(imports) for imports in train_df['imports']]
    test_df['corpus'] = [' '.join(imports) for imports in test_df['imports']]

    tfidf_vectorizer = TfidfVectorizer()
    x_train = tfidf_vectorizer.fit_transform(train_df['corpus'])
    x_test = tfidf_vectorizer.transform(test_df['corpus'])
    y_train = train_df['label']
    y_test = test_df['label']

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print('logistic regression model accuracy :', acc)

    return tfidf_vectorizer, model


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


def create_attention_model(train_df, test_df, epochs=2):
    batch_size = 512
    buffer_size = len(train_df.index)

    x_train = train_df['imports'].apply(lambda row: ' '.join(row)).to_list()
    x_test = test_df['imports'].apply(lambda row: ' '.join(row)).to_list()
    y_train = np.array(train_df['label'], dtype='float32')
    y_test = np.array(test_df['label'], dtype='float32')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(buffer_size).batch(batch_size)

    model = MyModel(train_dataset)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='tensorboard/{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[tensorboard])
    return model
