import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, Dropout, Embedding, Input, LSTM


def predict_one(df, model):
    x = df
    y_pred = model.predict(x)
    return y_pred


def predict_two(df, model):
    x = df['imports'].apply(lambda row: ' '.join(row)).to_list()
    y_pred = model.predict(x)
    return y_pred


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


def predict_with_attention_model(df, model):
    x_test = df['imports'].apply(lambda row: ' '.join(row)).to_list()
    y_pred = model.predict(x_test)
    return y_pred
