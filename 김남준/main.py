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

file_names = [path + 'test_features.jsonl', path + 'created_features.jsonl']
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
    n_pca=8
)

print('reduce features train 완료')
print(datetime.now(), '\n')

my.preprocessing.reduce_features_for_test(
    file_name=path + 'test_df.csv',
    props_file_path=path + 'reduce_features_props.pickle',
    save_path=path + 'test_pca_df.csv'
)

print('reduce features test 완료')
print(datetime.now(), '\n')

x_train = pd.read_csv(path + 'train_pca_df.csv')
x_test = pd.read_csv(path + 'test_pca_df.csv')
