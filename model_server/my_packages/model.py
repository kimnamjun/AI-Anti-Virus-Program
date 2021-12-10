import os
import numpy as np
import tensorflow as tf
from pycaret.classification import *

from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from tensorflow.keras import Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, Dropout, Embedding, Input, LSTM


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


def create_my_model(train_dataset):
    class BahdanauAttention(Model):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = Dense(units)
            self.W2 = Dense(units)
            self.V = Dense(1)

        def call(self, values, query):
            hidden_with_time_axis = tf.expand_dims(query, 1)
            score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector, attention_weights

    vectorize_layer = TextVectorization(output_mode='int', output_sequence_length=400)
    vectorize_layer.adapt(train_dataset.map(lambda text, label: text))
    sequence_input = Input(shape=(1,), dtype='string')
    vector_input = vectorize_layer(sequence_input)
    embedded_sequences = Embedding(len(vectorize_layer.get_vocabulary()), 128, mask_zero=True)(vector_input)
    lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(embedded_sequences)
    lstm, forward_h, forward_c, backward_h, backward_c \
        = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    attention = BahdanauAttention(64)
    context_vector, attention_weights = attention(lstm, state_h)
    dense1 = Dense(20, activation='relu')(context_vector)
    dropout = Dropout(0.5)(dense1)
    output = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=sequence_input, outputs=output)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    return model


def create_attention_model(train_df, test_df, epochs=2):
    batch_size = 512
    checkpoint_path = './checkpoint/checkpoint.ckpt'

    x_train = train_df['imports'].apply(lambda row: ' '.join(row)).to_list()
    x_test = test_df['imports'].apply(lambda row: ' '.join(row)).to_list()
    y_train = np.array(train_df['label'], dtype='float32')
    y_test = np.array(test_df['label'], dtype='float32')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    model = create_my_model(train_dataset)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[checkpoint])
    return model
