import my
from datetime import datetime

print('모델 생성 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

filenames = [f'train_features_{i}.jsonl' for i in range(6)]
train_df = my.preprocessing_one.convert_json_to_df(filenames)
train_df, props = my.preprocessing_one.reduce_features_for_train(train_df, n_pca=9)

filenames = ['test_features.jsonl']
test_df = my.preprocessing_one.convert_json_to_df(filenames)
test_df = my.preprocessing_one.reduce_features_for_test(test_df, props)

model = my.modeling_one.create_random_forest_model(train_df, test_df)

my.aws.save_to_s3(train_df, 'ava-data-csv', 'one/train_df.csv')
my.aws.save_to_s3(props, 'ava-data-model', 'one/properties.pickle')
my.aws.save_to_s3(model, 'ava-data-model', 'one/model.pickle')

print('모델 생성 종료', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
