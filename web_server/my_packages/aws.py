import io
import os
import csv
import boto3
import pickle
import shutil
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
    bucket = s3_resource.Bucket(name=bucket_name)
    for obj in bucket.objects.all():
        if obj.key.startswith(prefix):
            os.makedirs('./model/' + obj.key[len(prefix)+1:], exist_ok=True)
            bucket.download_file(obj.key, './model/' + obj.key[len(prefix)+1:])

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



def save_to_dynamo(filename_in_local: str, table_name: str):
    _, extension = os.path.splitext(filename_in_local)
    name = 'temp' + extension

    if extension == '.csv':
        table = ddb_resource.Table(table_name)
        with open(filename_in_local) as file, table.batch_writer() as batch:
            for row in csv.DictReader(file):
                batch.put_item(Item={
                    'sha256': row['sha256'],
                    'pca1': row['pca1'],
                    'pca2': row['pca2'],
                    'pca3': row['pca3'],
                    'pca4': row['pca4'],
                    'pca5': row['pca5'],
                    'pca6': row['pca6'],
                    'pca7': row['pca7'],
                    'pca8': row['pca8'],
                    'pca9': row['pca9'],
                    'label': row['label']
                })
    elif extension == '.pickle':
        table = ddb_resource.Table(table_name)
        with open(filename_in_local) as file, table.batch_writer() as batch:
            df = pickle.load(file)
            sha256 = df['sha256']
            imports = df['imports']
            for irow in range(len(df.index)):
                batch.put_item(Item={
                    'sha256': sha256[irow],
                    'imports': imports[irow]
                })
    else:
        raise FileNameException(msg='허용되지 않은 확장자입니다.')


class FileNameException(Exception):
    def __init__(self, msg='예상치 못한 예외 상황!'):
        self.msg = msg

    def __str__(self):
        return self.msg
