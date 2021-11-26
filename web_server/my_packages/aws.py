import io
import os
import csv
import boto3
import pickle

path = './dataset/temp/'
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
ddb_resource = boto3.resource('dynamodb')


def load_from_s3(filename: str, bucket_name: str):
    obj = s3_resource.Object(bucket_name, filename)
    file_obj = obj.get()['Body'].read()
    file = io.BytesIO(file_obj)
    content = pickle.load(file)
    return content


def save_to_s3(obj, bucket_name: str, filename: str):
    try:
        if filename.count('.') != 1:
            raise FileNameException()
        _, extension = os.path.splitext(filename)
        name = 'temp' + extension

        if extension == '.csv':
            obj.to_csv(path + name, header=False, index=False)
        elif extension == '.pickle':
            with open(path + name, 'wb') as file:
                pickle.dump(obj, file)

        s3_client.upload_file(path + name, bucket_name, filename)

    finally:
        if os.path.isfile(path + 'temp.csv'):
            os.remove(path + 'temp.csv')
        if os.path.isfile(path + 'temp.pickle'):
            os.remove(path + 'temp.pickle')


def save_to_dynamo(filename: str, table_name: str):
    table = ddb_resource.Table(table_name)
    with open(filename) as file, table.batch_writer() as batch:
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


class FileNameException(Exception):
    def __init__(self, msg='파일 이름을 확인하세요. (.)dot은 하나만 포함되어야 합니다.'):
        self.msg = msg

    def __str__(self):
        return self.msg
