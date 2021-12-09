import my
import os
from datetime import datetime

os.makedirs('./checkpoint/', exist_ok=True)
os.makedirs('./temp/', exist_ok=True)


def create_model(train_filenames, test_filenames):
    try:
        print('첫번째 모델 생성 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        train_df_one = my.preprocessing_one.convert_json_to_df(train_filenames)
        test_df_one = my.preprocessing_one.convert_json_to_df(test_filenames)
        train_df_one, props_one = my.preprocessing_one.reduce_features_for_train(train_df_one, n_pca=9)
        test_df_one = my.preprocessing_one.reduce_features_for_test(test_df_one, props_one)
        model_one = my.model.create_voting_model(train_df_one, test_df_one)

        my.aws.save_to_s3(train_df_one, 'ava-data-csv', 'one/train_df.csv')
        my.aws.save_to_s3(test_df_one, 'ava-data-csv', 'one/test_df.csv')
        my.aws.save_to_s3(props_one, 'ava-data-model', 'one/properties.pickle')
        my.aws.save_to_s3(model_one, 'ava-data-model', 'one/voting_model.pickle')

    except Exception as err:
        print('첫번째 모델 생성 실패', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        raise err

    else:
        print('첫번째 모델 저장 완료', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    train_filenames = [f'train_features_{i}.jsonl' for i in range(6)]
    test_filenames = ['test_features.jsonl']
    create_model(train_filenames, test_filenames)
