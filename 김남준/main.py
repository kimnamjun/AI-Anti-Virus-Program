from datetime import datetime
import pandas as pd
import my.preprocessing
import my.modeling

path = 'C:/Users/user/PycharmProjects/AVA/dataset/'

print('프로그램 실행')
print(datetime.now(), '\n')

file_names = ['C:/Windows/py.exe', 'C:/Windows/explorer.exe', 'C:/Users/user/Desktop/minigame.exe']
my.preprocessing.file2json(
    file_names=file_names,
    save_path=path + 'created_features.jsonl',
    print_result=2
)

print('file2json 완료')
print(datetime.now(), '\n')

file_names = [path + f'train_features_{i}.jsonl' for i in range(6)]
my.preprocessing.jsonl2df(
    file_names=file_names,
    save_path=path + 'train_df.csv'
)

print('json2df train 완료')
print(datetime.now(), '\n')

file_names = [path + 'test_features.jsonl']
my.preprocessing.jsonl2df(
    file_names=file_names,
    save_path=path + 'valid_df.csv'
)

print('json2df valid 완료')
print(datetime.now(), '\n')

file_names = [path + 'created_features.jsonl']
my.preprocessing.jsonl2df(
    file_names=file_names,
    save_path=path + 'test_df.csv'
)

print('json2df test 완료')
print(datetime.now(), '\n')

my.preprocessing.reduce_features_for_train(
    file_name=path + 'train_df.csv',
    save_path=path + 'train_pca_df.csv',
    save_path_props=path + 'reduce_features_props.pickle',
    n_pca=10
)

print('reduce features train 완료')
print(datetime.now(), '\n')

my.preprocessing.reduce_features_for_test(
    file_name=path + 'valid_df.csv',
    props_file_path=path + 'reduce_features_props.pickle',
    save_path=path + 'valid_pca_df.csv'
)

print('reduce features valid 완료')
print(datetime.now(), '\n')

my.preprocessing.reduce_features_for_test(
    file_name=path + 'test_df.csv',
    props_file_path=path + 'reduce_features_props.pickle',
    save_path=path + 'test_pca_df.csv'
)

print('reduce features test 완료')
print(datetime.now(), '\n')

train_df = pd.read_csv(path + 'train_pca_df.csv', index_col='sha256')
valid_df = pd.read_csv(path + 'valid_pca_df.csv', index_col='sha256')
test_df = pd.read_csv(path + 'test_pca_df.csv', index_col='sha256')

model = my.modeling.create_random_forest(train_df, valid_df, save_path=path + 'rf_model.pickle')

result = my.modeling.predict_with_random_forest(valid_df, model)

print('random forest model 생성 및 결과 예측 완료')
print(datetime.now(), '\n')
