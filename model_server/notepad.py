import io
import boto3

# S3 파일 확인하기
# s3 = boto3.client('s3')
# paginator = s3.get_paginator('list_objects_v2')
# response_iterator =  paginator.paginate(Bucket='ava-data-json')
# for page in response_iterator:
#     for content in page['Contents']:
#         print(content['Key'])

# S3에 저장하기
# s3 = boto3.client('s3')
# bucket_name = 'ava-data-json'
# file_name = 'test_features.jsonl'
# s3.upload_file('./temp/test_features.jsonl', bucket_name, file_name)

# S3에서 파일 읽기
# s3 = boto3.resource('s3')
# obj = s3.Object('ava-data-json', 'dataset/s3_test.txt')
# fileobj = obj.get()['Body'].read()
# file = io.BytesIO(fileobj)
# while True:
#    line = file.readline().decode('UTF-8')
#    if not line:
#        break
#    print(line)
