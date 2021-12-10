import io
import os
import csv
import boto3
import pickle

s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')
ddb_resource = boto3.resource('dynamodb')


def load_from_s3(filename: str, bucket_name: str):
    obj = s3_resource.Object(bucket_name, filename)
    file_obj = obj.get()['Body'].read()
    file = io.BytesIO(file_obj)
    content = pickle.load(file)
    return content


def load_weights_from_s3(model, bucket_name: str):
    checkpoint = s3_resource.Bucket(name=bucket_name)
    for obj in checkpoint.objects.all():
        path, filename = os.path.split(obj.key)
        if path.startswith('two/checkpoint'):
            checkpoint.download_file(obj.key, './checkpoint/' + filename)

    model.load_weights('./checkpoint/checkpoint')
    return model


def save_to_s3(obj, bucket_name: str, filename_in_s3: str):
    _, extension = os.path.splitext(filename_in_s3)
    name = './temp/temp' + extension

    if extension == '.csv':
        obj.to_csv(name, header=False, index=False)
    elif extension == '.pickle':
        with open(name, 'wb') as file:
            pickle.dump(obj, file)
    else:
        FileNameException(msg=f'이게 왜 안 돼! {extension}')

    s3_client.upload_file(name, bucket_name, filename_in_s3)


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
