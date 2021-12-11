import my
import numpy as np
import tensorflow as tf

train_df_two = my.aws.load_from_s3('two/train_df.pickle', 'ava-data-csv')
x_train = train_df_two['imports'].apply(lambda row: ' '.join(row)).to_list()
y_train = np.array(train_df_two['label'], dtype='float32')
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(512)
model_two = my.model.create_my_model(train_dataset)
model_two = my.aws.load_weights_from_s3(model_two, 'ava-data-model')

test_df_two = my.aws.load_from_s3('two/test_df.pickle', 'ava-data-csv')
x_test = test_df_two['imports'].apply(lambda row: ' '.join(row)).to_list()
y_test = np.array(test_df_two['label'], dtype='float32')
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(512)

model_two.evaluate(test_dataset)

y_pred = model_two.predict(x_test)
print(y_pred)
