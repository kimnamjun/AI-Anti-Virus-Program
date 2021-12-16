import io
import re
import json
import boto3
import pandas as pd
import statsmodels.api as sm
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

s3 = boto3.resource('s3')
pattern = re.compile(r'\W*(\w+)')


def convert_json_to_df(file_names: list):
    table1 = defaultdict(list)
    table2 = {'sha256': list(), 'label': list(), 'imports': list()}

    for file_name in file_names:
        obj = s3.Object('ava-data-json-main', file_name)
        file_obj = obj.get()['Body'].read()
        file = io.BytesIO(file_obj)
        while True:
            line = file.readline().decode('UTF-8')
            if not line:
                break
            line_json = json.loads(line)

            if line_json['label'] == -1:
                continue

            table1['label'].append(line_json['label'])
            table1['sha256'].append(line_json['sha256'])
            table1['g_exports'].append(line_json['general']['exports'])
            table1['g_has_debug'].append(line_json['general']['has_debug'])
            table1['g_has_relocations'].append(line_json['general']['has_relocations'])
            table1['g_has_resources'].append(line_json['general']['has_resources'])
            table1['g_has_signature'].append(line_json['general']['has_signature'])
            table1['g_has_tls'].append(line_json['general']['has_tls'])
            table1['g_imports'].append(line_json['general']['imports'])
            table1['g_size'].append(line_json['general']['size'])
            table1['g_symbols'].append(line_json['general']['symbols'])
            table1['g_vsize'].append(line_json['general']['vsize'])
            table1['h_major_image_version'].append(line_json['header']['optional']['major_image_version'])
            table1['h_major_linker_version'].append(line_json['header']['optional']['major_linker_version'])
            table1['h_major_operating_system_version'].append(line_json['header']['optional']['major_operating_system_version'])
            table1['h_major_subsystem_version'].append(line_json['header']['optional']['major_subsystem_version'])
            table1['h_minor_image_version'].append(line_json['header']['optional']['minor_image_version'])
            table1['h_minor_linker_version'].append(line_json['header']['optional']['minor_linker_version'])
            table1['h_minor_operating_system_version'].append(line_json['header']['optional']['minor_operating_system_version'])
            table1['h_minor_subsystem_version'].append(line_json['header']['optional']['minor_subsystem_version'])
            table1['h_sizeof_code'].append(line_json['header']['optional']['sizeof_code'])
            table1['h_sizeof_headers'].append(line_json['header']['optional']['sizeof_headers'])
            table1['h_sizeof_heap_commit'].append(line_json['header']['optional']['sizeof_heap_commit'])
            table1['h_timestamp'].append(line_json['header']['coff']['timestamp'])
            sec_entropy, sec_size = list(), list()
            for section in line_json['section']['sections']:
                sec_entropy.append(section['entropy'])
                sec_size.append(section['size'])
            table1['s_entropy'].append(1 if sum([0 if 0 < s < 7 else 1 for s in sec_entropy]) else 0)
            table1['s_size'].append(1 if sum([0 if s != 0 else 1 for s in sec_size]) else 0)

            table2['label'].append(line_json['label'])
            table2['sha256'].append(line_json['sha256'])
            imports = list()
            for key, value in line_json['imports'].items():
                for val in value:
                    res = re.match(pattern, val)
                    if res is None:
                        continue
                    val = res.group(1)
                    imports.append(val)
            table2['imports'].append(imports)

    return pd.DataFrame(table1), pd.DataFrame(table2)


def _get_variables_by_variable_selection(features_df: pd.DataFrame, label: pd.Series) -> list:
    variables = features_df.columns.tolist()
    y = label.tolist()
    selected_variables = list()

    sl_enter = 0.05
    sl_remove = 0.05

    while len(variables) > 0:
        remainder = list(set(variables) - set(selected_variables))
        pval = pd.Series(index=remainder)

        for col in remainder:
            X = features_df[selected_variables + [col]]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            pval[col] = model.pvalues[col]

        min_pval = pval.min()
        if min_pval < sl_enter:
            selected_variables.append(pval.idxmin())
            while len(selected_variables) > 0:
                selected_X = features_df[selected_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y, selected_X).fit().pvalues[1:]
                max_pval = selected_pval.max()
                if max_pval >= sl_remove:
                    remove_variable = selected_pval.idxmax()
                    selected_variables.remove(remove_variable)
                else:
                    break
        else:
            break

    return selected_variables


def reduce_features_for_train(df: pd.DataFrame, n_pca=9) -> (pd.DataFrame, list):
    df = df.set_index('sha256')
    features_df = df.drop('label', axis=1)
    label = df['label']

    scaler = StandardScaler()
    features_df = pd.DataFrame(scaler.fit_transform(features_df), index=df.index, columns=features_df.columns)

    selected_features = _get_variables_by_variable_selection(features_df, label)
    features_df = features_df[selected_features]

    pca = PCA(n_components=n_pca)
    pca_arr = pca.fit_transform(features_df, label)
    pca_df = pd.DataFrame(pca_arr, index=df.index, columns=[f'pca{i+1}' for i in range(n_pca)])
    pca_df['label'] = label.tolist()
    pca_df = pca_df.reset_index()

    reduce_features_props = selected_features, scaler, n_pca, pca
    return pca_df, reduce_features_props


def reduce_features_for_test(df: pd.DataFrame, props: tuple) -> pd.DataFrame:
    selected_features, scaler, n_pca, pca = props

    df = df.set_index('sha256')
    features_df = df.drop('label', axis=1)
    label = df['label']

    features_df = pd.DataFrame(scaler.transform(features_df), index=df.index, columns=features_df.columns)
    features_df = features_df[selected_features]

    pca_arr = pca.transform(features_df)
    pca_df = pd.DataFrame(pca_arr, index=df.index, columns=[f'pca{i+1}' for i in range(n_pca)])
    pca_df['label'] = label.tolist()
    pca_df = pca_df.reset_index()

    return pca_df

def preprocess_api(train_df, test_df, max_length=300) -> tuple:
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
