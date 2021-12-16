import os
import pickle
import boto3

path = './temp/'
s3 = boto3.client('s3')


def save_to_s3(obj, bucket_name: str, filename: str):
    _, extension = os.path.splitext(filename)
    basename = os.path.basename(filename)

    if extension == '.csv':
        obj.to_csv(path + basename, header=False, index=False)
        s3.upload_file(path + basename, bucket_name, filename)
    elif extension == '.pickle':
        with open(path + basename, 'wb') as file:
            pickle.dump(obj, file)
        s3.upload_file(path + basename, bucket_name, filename)
    elif basename.endswith('model'):
        obj.save(path + 'model')
        for dirpath, dirnames, filenames in os.walk(path + 'model/'):
            for dirname in dirnames:
                os.makedirs(os.path.join(dirpath, dirname), exist_ok=True)
            for filename in filenames:
                fn = os.path.join(dirpath, filename).replace('\\', '/')
                s3.upload_file(fn, bucket_name, 'two' + fn[6:])  # '/temp/' 제거


class FileNameException(Exception):
    def __init__(self, msg='예상치 못한 예외 상황!'):
        self.msg = msg

    def __str__(self):
        return self.msg
