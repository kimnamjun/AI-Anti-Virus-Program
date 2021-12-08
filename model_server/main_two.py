import my
from datetime import datetime


def create_model(train_filenames, test_filenames):
    try:
        print('두번째 모델 생성 시작', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        train_df_two = my.preprocessing_two.convert_json_to_df(train_filenames)
        test_df_two = my.preprocessing_two.convert_json_to_df(test_filenames)
        train_df_two, test_df_two, props_two = my.preprocessing_two.preprocess(train_df_two, test_df_two, max_length=389)
        model_two = my.model.create_attention_model(train_df_two, test_df_two)

        my.aws.save_to_s3(train_df_two, 'ava-data-csv', 'two/train_df.pickle')
        my.aws.save_to_s3(test_df_two, 'ava-data-csv', 'two/train_df.pickle')
        my.aws.save_to_s3(props_two, 'ava-data-model', 'two/properties.pickle')
        my.aws.save_to_s3(model_two, 'ava-data-model', 'two/model.h5')

    except Exception as err:
        print('두번째 모델 생성 실패', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        raise err

    else:
        print('두번째 모델 저장 완료', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    train_filenames = [f'train_features_{i}.jsonl' for i in range(6)]
    test_filenames = ['test_features.jsonl']
    create_model(train_filenames, test_filenames)
