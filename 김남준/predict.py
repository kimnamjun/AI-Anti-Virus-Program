import pickle
import my_package as my

path = 'C:/Users/user/PycharmProjects/AVA/dataset/'
train_file_names, test_file_names = my.util.load_file_names(path)
exe_file_names = ['C:/Windows/explorer.exe']

my.file2pe.file2pe(file_names=exe_file_names, save_path=path + 'created_jsonl/created_jsonl')

my.util.print_process('데이터전처리(one) 실행')
created_df = my.preprocessing_one.jsonl2df(file_names=[path + 'created_jsonl/created_jsonl'])
with open(path + 'properties/one_props.pickle', 'rb') as file:
    props = pickle.load(file)
print(created_df)
created_df = my.preprocessing_one.reduce_features_for_test(df=created_df, props=props)
my.util.save(path, 5, created_df)

with open(path + 'models/rf_model.pickle', 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(created_df)
print(y_pred)
