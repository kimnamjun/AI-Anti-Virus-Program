import io
import os
import boto3
import pickle
import shutil
from decimal import Decimal

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

path = './temp/'
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
ddb_resource = boto3.resource('dynamodb')


def load_pickle_from_s3(filename: str, bucket_name: str):
    obj = s3_resource.Object(bucket_name, filename)
    file_obj = obj.get()['Body'].read()
    file = io.BytesIO(file_obj)
    content = pickle.load(file)
    return content


def load_model_from_s3(prefix, bucket_name: str):
    shutil.rmtree('./model/')
    os.makedirs('./model/variables/', exist_ok=True)

    s3_client.download_file('ava-data-model-main', 'two/model/keras_metadata.pb', './model/keras_metadata.pb')
    s3_client.download_file('ava-data-model-main', 'two/model/saved_model.pb', './model/saved_model.pb')
    s3_client.download_file('ava-data-model-main', 'two/model/variables/variables.index',
                            './model/variables/variables.index')
    s3_client.download_file('ava-data-model-main', 'two/model/variables/variables.data-00000-of-00001',
                            './model/variables/variables.data-00000-of-00001')

    model = load_model('./model')
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    return model


def save_to_s3(obj, bucket_name: str, filename: str):
    if isinstance(obj, str):
        s3_client.upload_file(obj, bucket_name, filename)
        return

    _, extension = os.path.splitext(filename)
    basename = os.path.basename(filename)

    if extension == '.csv':
        obj.to_csv(path + basename, header=False, index=False)
        s3_client.upload_file(path + basename, bucket_name, filename)
    elif extension == '.pickle':
        with open(path + basename, 'wb') as file:
            pickle.dump(obj, file)
        s3_client.upload_file(path + basename, bucket_name, filename)
    elif basename.endswith('model'):
        obj.save(path + 'model')
        for dirpath, dirnames, filenames in os.walk(path + 'model/'):
            for dirname in dirnames:
                os.makedirs(os.path.join(dirpath, dirname), exist_ok=True)
            for filename in filenames:
                fn = os.path.join(dirpath, filename).replace('\\', '/')
                s3_client.upload_file(fn, bucket_name, 'two' + fn[6:])  # '/temp/' 제거


def save_to_dynamo(df: pd.DataFrame, table_name: str):
    table = ddb_resource.Table(table_name)

    if table_name == 'AVA-01':
        sha256, label = df['sha256'].tolist(), df['label'].tolist()
        pca1, pca2, pca3 = df['pca1'].tolist(), df['pca2'].tolist(), df['pca3'].tolist()
        pca4, pca5, pca6 = df['pca4'].tolist(), df['pca5'].tolist(), df['pca6'].tolist()
        pca7, pca8, pca9 = df['pca7'].tolist(), df['pca8'].tolist(), df['pca9'].tolist()

        with table.batch_writer() as batch:
            for i in range(len(df.index)):
                batch.put_item(Item={
                    'sha256': sha256[i],
                    'pca1': Decimal(str(pca1[i])),
                    'pca2': Decimal(str(pca2[i])),
                    'pca3': Decimal(str(pca3[i])),
                    'pca4': Decimal(str(pca4[i])),
                    'pca5': Decimal(str(pca5[i])),
                    'pca6': Decimal(str(pca6[i])),
                    'pca7': Decimal(str(pca7[i])),
                    'pca8': Decimal(str(pca8[i])),
                    'pca9': Decimal(str(pca9[i])),
                    'label': label[i]
                })

    elif table_name == 'AVA-02':
        sha256, label = df['sha256'].tolist(), df['label'].tolist()
        imports = df['imports'].apply(lambda row: ' '.join(row)).to_list()

        with table.batch_writer() as batch:
            for i in range(len(df.index)):
                batch.put_item(Item={
                    'sha256': sha256[i],
                    'imports': imports[i],
                    'label': label[i]
                })
