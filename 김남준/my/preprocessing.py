import json
import pandas as pd
import statsmodels.api as sm
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def jsonl2df(file_names: list) -> pd.DataFrame:
    """
    jsonl 파일에서 특정 컬럼을 추출하여 pd.DataFrame 형태로 변환합니다.
    :param file_names: 경로를 포함한 jsonl 파일 목록입니다.
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


def reduce_features_for_train(df: pd.DataFrame, save_path: str, n_pca=9) -> (pd.DataFrame, list):
    """
    변수 선택법과 PCA를 이용하여 features의 개수를 줄입니다.
    train data를 위한 함수입니다.
    :param df: 분석에 필요한 features를 DataFrame 형태로 추출한 데이터입니다.
    :param save_path: csv 파일을 저장할 경로입니다.
    :param n_pca: PCA components 개수를 결정합니다.
    :return: 생성된 pca_df와 properties입니다.
    """
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

    return pca_df, reduce_features_props


def reduce_features_for_test(df: pd.DataFrame, props: tuple, save_path: str) -> pd.DataFrame:
    """
    변수 선택법과 PCA를 이용하여 features의 개수를 줄입니다.
    test data를 위한 함수이며 reduce_features_for_train이 선행되어야 합니다.
    :param df: 분석에 필요한 features를 DataFrame 형태로 추출한 데이터입니다.
    :param props: reduce_features_for_train에서 생성된 properties입니다.
    :param save_path: csv 파일을 저장할 경로입니다.
    :return: 생성된 pca_df입니다.
    """
    selected_features, scaler, n_pca, pca = props

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
