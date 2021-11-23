import json
import pandas as pd
from collections import defaultdict


def json2df(file_name: str) -> pd.DataFrame:
    table = defaultdict(list)

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

    return pd.DataFrame(table)


def reduce_features(df: pd.DataFrame, props: tuple) -> pd.DataFrame:
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
