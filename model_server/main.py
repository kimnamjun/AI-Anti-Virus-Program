import my
import os
from datetime import datetime

os.makedirs('./temp/', exist_ok=True)

train_filenames = [f'train_features_{i}.jsonl' for i in range(6)]
test_filenames = ['test_features.jsonl']

try:
    print('전처리 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    train_df_one, train_df_two = my.preprocess.convert_json_to_df(train_filenames)
    test_df_one, test_df_two = my.preprocess.convert_json_to_df(test_filenames)

    print('전처리 완료\n첫번째 모델 생성 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    train_df_one, props_one = my.preprocess.reduce_features_for_train(train_df_one, n_pca=9)
    test_df_one = my.preprocess.reduce_features_for_test(test_df_one, props_one)
    model_one = my.model.create_voting_model(train_df_one, test_df_one)

    my.aws.save_to_s3(train_df_one, 'ava-data-csv-main', 'one/train_df.csv')
    my.aws.save_to_s3(test_df_one, 'ava-data-csv-main', 'one/test_df.csv')
    my.aws.save_to_s3(props_one, 'ava-data-model-main', 'one/properties.pickle')
    my.aws.save_to_s3(model_one, 'ava-data-model-main', 'one/voting_model.pickle')

    print('첫번째 모델 저장 완료\n두번째 모델 생성 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    train_df_two, test_df_two, props_two = my.preprocess.preprocess_api(train_df_two, test_df_two, max_length=389)
    model_two = my.model.create_attention_model(train_df_two, test_df_two, epochs=2)

    my.aws.save_to_s3(train_df_two, 'ava-data-csv-main', 'two/train_df.pickle')
    my.aws.save_to_s3(test_df_two, 'ava-data-csv-main', 'two/test_df.pickle')
    my.aws.save_to_s3(props_two, 'ava-data-model-main', 'two/properties.pickle')
    my.aws.save_to_s3(model_two, 'ava-data-model-main', 'two/model')

    print('두번째 모델 저장 완료', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

except Exception as err:
    print('모델 생성 실패', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    raise err
else:
    print('모델 생성 완료', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
