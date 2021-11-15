from datetime import datetime
import pandas as pd
import my.file2pe
import my.preprocessing
import my.modeling

def print_process(x):
    print(x, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '', sep='\n')

path = 'C:/Users/user/PycharmProjects/AVA/dataset/'

print_process('프로그램 실행')

file_names = [path + 'NJ/' + f for f in ['always on top.exe', 'explorer.exe', 'minigame.exe', 'notepad.exe', 'py.exe']]
my.file2pe.file2pe(file_names=file_names, save_path=path + 'NJ/created_features.jsonl', verbose=2)

print_process('file2json 완료 \njson2df 실행')

train_file_names = [path + f'train_features_{i}.jsonl' for i in range(6)]
valid_file_names = [path + 'test_features.jsonl']
test_file_names = [path + 'NJ/created_features.jsonl']

train_df = my.preprocessing.jsonl2df(file_names=train_file_names)
valid_df = my.preprocessing.jsonl2df(file_names=valid_file_names)
test_df = my.preprocessing.jsonl2df(file_names=test_file_names)

print_process('json2df 완료 \n reduce features train 실행')

train_pca_df, props = my.preprocessing.reduce_features_for_train(df=train_df, save_path=path + 'NJ/train_pca_df.csv', n_pca=9)
valid_pca_df = my.preprocessing.reduce_features_for_test(df=valid_df, props=props, save_path=path + 'NJ/valid_pca_df.csv')
test_pca_df = my.preprocessing.reduce_features_for_test(df=test_df, props=props, save_path=path + 'NJ/test_pca_df.csv')

print_process('reduce features 완료 \n모델 생성 실행')

train_df = pd.read_csv(path + 'NJ/train_pca_df.csv', index_col='sha256')
valid_df = pd.read_csv(path + 'NJ/valid_pca_df.csv', index_col='sha256')
test_df = pd.read_csv(path + 'NJ/test_pca_df.csv', index_col='sha256')

model = my.modeling.create_random_forest(
    train_df=train_df,
    test_df=valid_df,
    save_path=path + 'NJ/rf_model.pickle'
)

print_process('모델 생성 완료 \n예측 실행')

result_valid = my.modeling.predict_with_model(valid_df, model)
result_test = my.modeling.predict_with_model(test_df, model)

print_process('예측 완료 \n프로그램 종료')
