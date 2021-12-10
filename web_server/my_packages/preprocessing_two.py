import io
import re
import json
import boto3
import pandas as pd

s3 = boto3.resource('s3')
pattern = re.compile(r'\W*(\w+)')


def preprocess(file_name: str, props: tuple) -> pd.DataFrame:
    max_length, functions = props
    table = {'sha256': list(), 'label': list(), 'imports': list()}

    with open(file_name) as file:
        while True:
            line = file.readline().decode('UTF-8')
            if not line:
                break
            line_json = json.loads(line)

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

        df = pd.DataFrame(table)
        imports = list()
        for idx, func_list in enumerate(df['imports']):
            func_list = [func for func in func_list if func in functions]
            func_list += [''] * (max_length - len(func_list))
            imports.append(func_list)
        df['imports'] = imports

    return df
