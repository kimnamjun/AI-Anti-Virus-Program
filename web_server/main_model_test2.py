import my
import os
import pickle
from datetime import datetime

os.makedirs('./temp/', exist_ok=True)
os.makedirs('./model/', exist_ok=True)

tm = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

filename = 'always_on_top.exe'
filename_path = './temp/always_on_top.exe'

json_file = my.file2pe.convert_file_to_pe(filename_path)
json_filename = f'_{filename}_{tm}'.replace('.', '_') + '.json'
with open(f'./temp/{json_filename}', 'w') as file:
    file.write(json_file)
    my.aws.save_to_s3(file, 'ava-data-json-main', json_filename)

df1, df2 = my.preprocess.convert_json_to_df(json_filename)

props_two = my.aws.load_pickle_from_s3('two/properties.pickle', 'ava-data-model-main')

df2 = my.preprocess.preprocess_api(df2, props_two)
with open('./temp/df_two.pickle', 'wb') as file:
    pickle.dump(df2, file)

print('모델 부르는 중')
model_two = my.aws.load_model_from_s3('two/model', 'ava-data-model-main')
print('모델 부르기 끝')

result2 = my.model.predict_two(df2, model_two)

print(f'RESULT is {result2}')