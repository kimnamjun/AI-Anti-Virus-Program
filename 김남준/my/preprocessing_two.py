import re
import json
import pandas as pd

pattern = re.compile(r'\W*(\w+)')


def jsonl2df(file_names: list) -> pd.DataFrame:
    table = {'sha256': list(), 'label': list(), 'imports': list()}
    for file_name in file_names:
        with open(file_name) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line_json = json.loads(line)

                if line_json['label'] == -1:
                    continue
                table['sha256'].append(line_json['sha256'])
                table['label'].append(line_json['label'])

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


# props = max_length, functions
def set_max_length_for_train(train_df: pd.DataFrame, test_df: pd.DataFrame, max_length: int) -> tuple:
    # handle outliers
    train_df = train_df[train_df['imports'].map(lambda x: len(x)) <= max_length].reset_index(drop=True)
    test_df = test_df[test_df['imports'].map(lambda x: len(x)) <= max_length].reset_index(drop=True)


    # extract functions
    functions = set()

    for function_list in train_df['imports']:
        for function in function_list:
            functions.add(function)

    imports = list()
    for idx, function_list in enumerate(test_df['imports']):
        function_list = [function for function in function_list if function in functions]
        imports.append(function_list)
    test_df['imports'] = imports


    # pad blank string
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

def set_max_length_for_test(df: pd.DataFrame, props: tuple) -> tuple:
    max_length, functions = props
    df = df[df['imports'].map(lambda x: len(x)) <= max_length].reset_index(drop=True)

    imports = list()
    for idx, function_list in enumerate(df['imports']):
        function_list = [function for function in function_list if function in functions]
        imports.append(function_list)
    df['imports'] = imports

    imports = list()
    for idx, func_list in enumerate(df['imports']):
        imports.append(func_list + [''] * (max_length - len(func_list)))
    df['imports'] = imports

    return df
