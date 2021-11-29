import my
from datetime import datetime

print('모델 생성 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

train_filenames = [f'train_features_{i}.jsonl' for i in range(6)]
test_filenames = ['test_features.jsonl']

train_df_one = my.preprocessing_one.convert_json_to_df(train_filenames)
test_df_one = my.preprocessing_one.convert_json_to_df(test_filenames)
train_df_one, props_one = my.preprocessing_one.reduce_features_for_train(train_df_one, n_pca=9)
test_df_one = my.preprocessing_one.reduce_features_for_test(test_df_one, props_one)
model_one = my.modeling_one.create_random_forest_model(train_df_one, test_df_one)

my.aws.save_to_s3(train_df_one, 'ava-data-csv', 'one/train_df.csv')
my.aws.save_to_s3(test_df_one, 'ava-data-csv', 'one/test_df.csv')
my.aws.save_to_s3(props_one, 'ava-data-model', 'one/properties.pickle')
my.aws.save_to_s3(model_one, 'ava-data-model', 'one/model.pickle')

train_df_two = my.preprocessing_two.convert_json_to_df(train_filenames)
test_df_two = my.preprocessing_two.convert_json_to_df(test_filenames)
train_df_two, test_df_two, props_two = my.preprocessing_two.preprocess(train_df_two, test_df_two, max_length=300)
model_two = my.modeling_two.create_model_with_tfidf_and_logistic_regression(train_df_two, test_df_two)

my.aws.save_to_s3(train_df_two, 'ava-data-csv', 'two/train_df.pickle')
my.aws.save_to_s3(test_df_two, 'ava-data-csv', 'two/train_df.pickle')
my.aws.save_to_s3(props_two, 'ava-data-model', 'two/properties.pickle')
my.aws.save_to_s3(model_two, 'ava-data-model', 'two/model.pickle')

print('모델 생성 종료', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
