import os
import pickle
import boto3

path = './temp/'
s3 = boto3.client('s3')


def save_to_s3(obj, bucket_name: str, filename: str):
    try:
        _, extension = os.path.splitext(filename)
        name = 'temp' + extension

        if extension == '.csv':
            obj.to_csv(path + name, header=False, index=False)
        elif extension == '.pickle':
            with open(path + name, 'wb') as file:
                pickle.dump(obj, file)

        s3.upload_file(path + name, bucket_name, filename)

    finally:
        if os.path.isfile(path + 'temp.csv'):
            os.remove(path + 'temp.csv')
        if os.path.isfile(path + 'temp.pickle'):
            os.remove(path + 'temp.pickle')


def save_weights_to_s3(path: str, bucket_name: str, file_dir: str):
    for filename in os.listdir(path):
        s3.upload_file(path + filename, bucket_name, file_dir + filename)


class FileNameException(Exception):
    def __init__(self, msg='예상치 못한 예외 상황!'):
        self.msg = msg

    def __str__(self):
        return self.msg
