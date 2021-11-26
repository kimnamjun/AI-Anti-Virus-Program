import os
import pickle
import boto3


path = './dataset/temp/'
s3 = boto3.client('s3')


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

        s3.upload_file(path + name, bucket_name, filename)

    finally:
        if os.path.isfile(path + 'temp.csv'):
            os.remove(path + 'temp.csv')
        if os.path.isfile(path + 'temp.pickle'):
            os.remove(path + 'temp.pickle')


class FileNameException(Exception):
    def __init__(self, msg='파일 이름을 확인하세요. (.)dot은 하나만 포함되어야 합니다.'):
        self.msg = msg

    def __str__(self):
        return self.msg
