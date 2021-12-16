import my
import os
import pickle
from datetime import datetime

os.makedirs('./temp/', exist_ok=True)
os.makedirs('./model/', exist_ok=True)

props_one = my.aws.load_pickle_from_s3('one/properties.pickle', 'ava-data-model-main')
model_one = my.aws.load_pickle_from_s3('one/voting_model.pickle', 'ava-data-model-main')

tm = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

filename_path = './temp/always_on_top.exe'
json_file = my.file2pe.convert_file_to_pe(filename_path)
if not json_file:
    print('변환 실패 에러')
    raise ZeroDivisionError

json_file_name = './temp/temp.json'
with open(json_file_name, 'w') as file:
    file.write(json_file)

df1, df2 = my.preprocess.convert_json_to_df(json_file_name)

df1 = my.preprocess.reduce_features(df1, props_one)
df1.to_csv('./temp/df_one.csv', index=False)

x1 = df1.drop(['sha256', 'label'], axis=1)
result1 = my.model.predict_one(x1, model_one)

print('RESULT :', result1)