from datetime import datetime
import pandas as pd
import my.preprocessing
import my.modeling

def print_process(x):
    print(x, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '', sep='\n')

path = 'C:/Users/user/PycharmProjects/AVA/dataset/'

print_process('프로그램 실행')

file_names = [path + 'NJ/' + f for f in ['always on top.exe', 'explorer.exe', 'minigame.exe', 'notepad.exe', 'py.exe']]
my.preprocessing.file2json(
    file_names=file_names,
    save_path=path + 'NJ/created_features.jsonl',
    print_result=2
)

print_process('file2json 완료 \njson2df train 실행')

file_names = [path + f'train_features_{i}.jsonl' for i in range(6)]
my.preprocessing.jsonl2df(
    file_names=file_names,
    save_path=path + 'NJ/train_df.csv'
)

print_process('json2df train 완료 \njson2df valid 실행')

file_names = [path + 'test_features.jsonl']
my.preprocessing.jsonl2df(
    file_names=file_names,
    save_path=path + 'NJ/valid_df.csv'
)

print_process('json2df valid 완료 \njson2df test 실행')

file_names = [path + 'NJ/created_features.jsonl']
my.preprocessing.jsonl2df(
    file_names=file_names,
    save_path=path + 'NJ/test_df.csv'
)

print_process('json2df test 완료 \njson2df 완료 \n reduce features train 실행')

my.preprocessing.reduce_features_for_train(
    file_name=path + 'NJ/train_df.csv',
    save_path=path + 'NJ/train_pca_df.csv',
    save_path_props=path + 'NJ/reduce_features_props.pickle',
    n_pca=10
)

print_process('reduce features train 완료 \nreduce features valid 실행')

my.preprocessing.reduce_features_for_test(
    file_name=path + 'NJ/valid_df.csv',
    props_file_path=path + 'NJ/reduce_features_props.pickle',
    save_path=path + 'NJ/valid_pca_df.csv'
)

print_process('reduce features valid 완료 \nreduce features test 실행')

my.preprocessing.reduce_features_for_test(
    file_name=path + 'NJ/test_df.csv',
    props_file_path=path + 'NJ/reduce_features_props.pickle',
    save_path=path + 'NJ/test_pca_df.csv'
)

print_process('reduce features test 완료 \nreduce features 완료 \n모델 생성 실행')

train_df = pd.read_csv(path + 'NJ/train_pca_df.csv', index_col='sha256')
valid_df = pd.read_csv(path + 'NJ/valid_pca_df.csv', index_col='sha256')
test_df = pd.read_csv(path + 'NJ/test_pca_df.csv', index_col='sha256')

model = my.modeling.create_random_forest(
    train_df=train_df,
    test_df=valid_df,
    save_path=path + 'NJ/rf_model.pickle'
)

print_process('모델 생성 완료 \n예측 실행')

result = my.modeling.predict_with_random_forest(valid_df, model)

print_process('예측 완료 \n프로그램 종료')

result