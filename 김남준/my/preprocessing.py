"""
각 함수마다 save_path가 유효한지 확인해봐야 할 듯
"""
import os
import json
import lief
import pickle
import hashlib
import pandas as pd
import statsmodels.api as sm
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def file2json(file_names: list, save_path: str, print_result=1) -> list:
    """
    파일 이름 목록을 넣으면 수집한 데이터와 비슷한 json 형태로 변환합니다.
    :param file_names: 경로를 포함한 실행 파일 목록입니다.
    :param save_path: save_path 지정 시 생성한 json 파일들을 jsonl 포맷으로 저장합니다.
    :param print_result: {0: 출력 안함, 1: 개수만 출력, 2: 파일 리스트 출력}
    :return: json 파일의 리스트입니다.
    """
    json_files = list()
    success_list = list()
    failure_list = list()

    for file_name in file_names:
        try:
            js = {'label': -1, 'datadirectories': [], 'general': {}, 'header': {'coff': {}, 'optional':{}},
                  'imports': {}, 'section': {'sections': []}}

            binary = lief.parse(file_name)

            for data_directory in binary.data_directories:
                dic = {'name': data_directory.type.name,
                       'size': data_directory.size,
                       'virtual_address': data_directory.rva}
                js['datadirectories'].append(dic)

            js['exports'] = binary.exported_functions

            js['general']['exports'] = len(binary.exported_functions)
            js['general']['has_debug'] = int(binary.has_debug)
            js['general']['has_relocations'] = int(binary.has_relocations)
            js['general']['has_resources'] = int(binary.has_resources)
            js['general']['has_signature'] = int(binary.has_signatures)
            js['general']['has_tls'] = int(binary.has_tls)
            js['general']['imports'] = len(binary.imports)
            js['general']['size'] = os.path.getsize(file_name)
            js['general']['symbols'] = len(binary.symbols)
            js['general']['vsize'] = binary.virtual_size

            js['header']['coff']['characteristics'] = [x.name for x in binary.header.characteristics_list]
            js['header']['coff']['machine'] = binary.header.machine.name
            js['header']['coff']['timestamp'] = binary.header.time_date_stamps

            js['header']['optional']['dll_characteristics'] = [x.name for x in binary.optional_header.dll_characteristics_lists]
            js['header']['optional']['magic'] = binary.optional_header.magic.name
            js['header']['optional']['major_image_version'] = binary.optional_header.major_image_version
            js['header']['optional']['major_linker_version'] = binary.optional_header.major_linker_version
            js['header']['optional']['major_operating_system_version'] = binary.optional_header.major_operating_system_version
            js['header']['optional']['major_subsystem_version'] = binary.optional_header.major_subsystem_version
            js['header']['optional']['minor_image_version'] = binary.optional_header.minor_image_version
            js['header']['optional']['minor_linker_version'] = binary.optional_header.minor_linker_version
            js['header']['optional']['minor_operating_system_version'] = binary.optional_header.minor_operating_system_version
            js['header']['optional']['minor_subsystem_version'] = binary.optional_header.minor_subsystem_version
            js['header']['optional']['sizeof_code'] = binary.optional_header.sizeof_code
            js['header']['optional']['sizeof_headers'] = binary.optional_header.sizeof_headers
            js['header']['optional']['sizeof_heap_commit'] = binary.optional_header.sizeof_heap_commit
            js['header']['optional']['subsystem'] = binary.optional_header.subsystem.name

            for imp_lib in binary.imports:
                dic = dict()
                dic[imp_lib.name] = [func.name for func in imp_lib.entries]
                js['imports'].update(dic)

            js['name'] = binary.name

            js['section']['entry'] = binary.sections[0].name
            for section in binary.sections:
                dic = {'name': section.name,
                       'entropy': section.entropy,
                       'props': [x.name for x in section.characteristics_lists],
                       'size': section.size,
                       'vsize': section.virtual_size}
                js['section']['sections'].append(dic)

            with open(file_name, 'rb') as file:
                js['sha256'] = hashlib.sha256(file.read()).hexdigest()

            json_files.append(json.dumps(js, sort_keys=True))
            success_list.append(file_name)

        except Exception as err:
            failure_list.append(file_name)

    with open(save_path, 'w') as file:
        for json_file in json_files:
            file.write(json_file)
            file.write('\n')
    print(save_path, '에 jsonl 파일이 저장되었습니다.')

    if print_result == 1:
        print(f'성공 {len(success_list)}개')
        print(f'실패 {len(failure_list)}개')
    elif print_result == 2:
        print(f'성공 {len(success_list)}개', success_list)
        print(f'실패 {len(failure_list)}개', failure_list)

    return json_files


def jsonl2df(file_names: list, save_path: str) -> pd.DataFrame:
    """
    jsonl 파일에서 특정 컬럼을 추출하여 pd.DataFrame 형태로 변환합니다.
    :param file_names: 경로를 포함한 jsonl 파일 목록입니다.
    :param save_path: save_path 지정 시 생성한 DataFrame을 csv 포맷으로 저장합니다.
    :return: 변환한 DataFrame입니다.
    """
    table = defaultdict(list)

    for file_name in file_names:
        with open(file_name) as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line_json = json.loads(line)

                table['sha256'].append(line_json['sha256'])
                table['label'].append(line_json['label'])

                table['g_exports'].append(line_json['general']['exports'])
                table['g_has_debug'].append(line_json['general']['has_debug'])
                table['g_has_relocations'].append(line_json['general']['has_relocations'])
                table['g_has_resources'].append(line_json['general']['has_resources'])
                table['g_has_signature'].append(line_json['general']['has_signature'])
                table['g_has_tls'].append(line_json['general']['has_tls'])
                table['g_imports'].append(line_json['general']['imports'])
                table['g_size'].append(line_json['general']['size'])
                table['g_symbols'].append(line_json['general']['symbols'])
                table['g_vsize'].append(line_json['general']['vsize'])

                table['h_major_image_version'].append(line_json['header']['optional']['major_image_version'])
                table['h_major_linker_version'].append(line_json['header']['optional']['major_linker_version'])
                table['h_major_operating_system_version'].append(line_json['header']['optional']['major_operating_system_version'])
                table['h_major_subsystem_version'].append(line_json['header']['optional']['major_subsystem_version'])
                table['h_minor_image_version'].append(line_json['header']['optional']['minor_image_version'])
                table['h_minor_linker_version'].append(line_json['header']['optional']['minor_linker_version'])
                table['h_minor_operating_system_version'].append(line_json['header']['optional']['minor_operating_system_version'])
                table['h_minor_subsystem_version'].append(line_json['header']['optional']['minor_subsystem_version'])
                table['h_sizeof_code'].append(line_json['header']['optional']['sizeof_code'])
                table['h_sizeof_headers'].append(line_json['header']['optional']['sizeof_headers'])
                table['h_sizeof_heap_commit'].append(line_json['header']['optional']['sizeof_heap_commit'])
                table['h_timestamp'].append(line_json['header']['coff']['timestamp'])

    df = pd.DataFrame(table)
    df.to_csv(save_path, index=False)
    print(save_path, '에 csv 파일이 저장되었습니다.')

    return df


def _get_variables_by_variable_selection(features_df: pd.DataFrame, label: pd.Series) -> list:
    """
    전진 단계별 선택법
    참고 : https://zephyrus1111.tistory.com/65
    :param features_df: x data
    :param label: y data
    :return: selected variables
    """
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


def reduce_features_for_train(file_name: str, save_path: str, save_path_props: str, n_pca=10) -> pd.DataFrame:
    """
    변수 선택법과 PCA를 이용하여 features의 개수를 줄입니다.
    train data를 위한 함수입니다.
    :param file_name: json2df를 통해 생성된 csv 파일의 경로입니다.
    :param save_path: csv 파일을 저장할 경로입니다.
    :param save_path_props: properties 파일을 저장할 경로입니다.
    :param n_pca: PCA components 개수를 결정합니다.
    :return: 생성된 pca_df입니다.
    """
    df = pd.read_csv(file_name)
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

    reduce_features_props = [selected_features, scaler, n_pca, pca]

    pca_df.to_csv(save_path)
    print(save_path, '에 csv 파일이 저장되었습니다.')

    with open(save_path_props, 'wb') as file:
        pickle.dump(reduce_features_props, file)
        print(save_path_props, '에 properties(pickle) 파일이 저장되었습니다.')

    return pca_df


def reduce_features_for_test(file_name: str, props_file_path: str, save_path: str) -> pd.DataFrame:
    """
    변수 선택법과 PCA를 이용하여 features의 개수를 줄입니다.
    test data를 위한 함수이며 reduce_features_for_train이 선행되어야 합니다.
    :param file_name: json2df를 통해 생성된 csv 파일의 경로입니다.
    :param props_file_path: reduce_features_for_train에서 생성된 properties 파일의 경로입니다.
    :return: 생성된 pca_df입니다.
    """
    with open(props_file_path, 'rb') as file:
        selected_features, scaler, n_pca, pca = pickle.load(file)

    df = pd.read_csv(file_name)
    df = df.set_index('sha256')
    features_df = df.drop('label', axis=1)
    label = df['label']

    features_df = pd.DataFrame(scaler.transform(features_df), index=df.index, columns=features_df.columns)
    features_df = features_df[selected_features]

    pca_arr = pca.transform(features_df)
    pca_df = pd.DataFrame(pca_arr, index=df.index, columns=[f'pca{i+1}' for i in range(n_pca)])
    pca_df['label'] = label.tolist()

    pca_df.to_csv(save_path)
    print(save_path, '에 csv 파일이 저장되었습니다.')

    return pca_df
