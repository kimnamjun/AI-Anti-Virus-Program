import my
import os
import pickle
from datetime import datetime

os.makedirs('./temp/', exist_ok=True)
os.makedirs('./model/', exist_ok=True)

props_one = my.aws.load_pickle_from_s3('one/properties.pickle', 'ava-data-model-main')
model_one = my.aws.load_pickle_from_s3('one/voting_model.pickle', 'ava-data-model-main')

tm = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

filename = 'always_on_top.exe'
filename_path = './temp/always_on_top.exe'

json_file = my.file2pe.convert_file_to_pe(filename_path)
json_filename = f'_{filename}_{tm}'.replace('.', '_') + '.json'
with open(f'./temp/{json_filename}', 'w') as file:
    file.write(json_file)
my.aws.save_to_s3(f'./temp/{json_filename}', 'ava-data-json-main', json_filename)

df1, df2 = my.preprocess.convert_json_to_df(f'_{filename}_{tm}.json')

df1 = my.preprocess.reduce_features(df1, props_one)
df1.to_csv('./temp/df_one.csv', index=False)

x1 = df1.drop(['sha256', 'label'], axis=1)
result1 = my.model.predict_one(x1, model_one)
