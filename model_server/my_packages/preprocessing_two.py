import io
import re
import json
import boto3
import pandas as pd

s3 = boto3.resource('s3')
pattern = re.compile(r'\W*(\w+)')


def convert_json_to_df(file_names: list) -> pd.DataFrame:
    """
    s3 안의 jsonl 파일에서 특정 컬럼을 추출하여 pd.DataFrame 형태로 변환합니다.
    :param file_names: 경로를 포함한 jsonl 파일 목록입니다.
    :return: 변환한 DataFrame입니다.
    """
    table = {'sha256': list(), 'label': list(), 'imports': list()}

    for file_name in file_names:
        obj = s3.Object('ava-data-json', file_name)
        file_obj = obj.get()['Body'].read()
        file = io.BytesIO(file_obj)
        while True:
            line = file.readline().decode('UTF-8')
            if not line:
                break
            line_json = json.loads(line)

            if line_json['label'] == -1:
                continue
            table['label'].append(line_json['label'])
            table['sha256'].append(line_json['sha256'])

            imports = list()
            for key, value in line_json['imports'].items():
                for val in value:
                    res = re.match(pattern, val)
                    if res is None:
                        continue
                    val = res.group(1)
                    imports.append(val)
            table['imports'].append(imports)

    return pd.DataFrame(table)


def preprocess(train_df, test_df, max_length=300) -> tuple:
    """
    API 함수 관련하여 데이터 전처리를 수행합니다.
    :param train_df: 학습용 데이터입니다.
    :param test_df: 검증용 데이터입니다.
    :param max_length: 하나의 파일이 가질 수 있는 최대의 API 함수 개수입니다.
    :return: 변환된 학습용 데이터, 검증용 데이터와 properties입니다.
    """
    # max_length 이상의 함수를 갖는 행 버리기
    train_df = train_df[train_df['imports'].map(lambda x: len(x)) <= max_length]
    test_df = test_df[test_df['imports'].map(lambda x: len(x)) <= max_length]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # 함수 추출 및 학습용 데이터에 들어있지 않은 함수 제거
    functions = set()
    for func_list in train_df['imports']:
        for func in func_list:
            functions.add(func)

    imports = list()
    for idx, func_list in enumerate(test_df['imports']):
        func_list = [func for func in func_list if func in functions]
        imports.append(func_list)
    test_df['imports'] = imports

    # 같은 크기를 가지도록 빈 문자열 패딩
    imports = list()
    for idx, func_list in enumerate(train_df['imports']):
        imports.append(func_list + [''] * (max_length - len(func_list)))
    train_df['imports'] = imports

    imports = list()
    for idx, func_list in enumerate(test_df['imports']):
        imports.append(func_list + [''] * (max_length - len(func_list)))
    test_df['imports'] = imports

    props = max_length, functions
    return train_df, test_df, props
