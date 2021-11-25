import io
import boto3
import pickle


s3c = boto3.client('s3')
s3r = boto3.resource('s3')


def load_from_s3(filename: str, bucket: str):
    obj = s3r.Object(bucket, filename)
    file_obj = obj.get()['Body'].read()
    file = io.BytesIO(file_obj)
    content = pickle.load(file)
    return content
