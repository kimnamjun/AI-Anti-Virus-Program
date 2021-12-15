import boto3

s3 = boto3.client('s3')

buckets = ['ava-data-model-main', 'ava-data-json-main', 'ava-data-csv-main']
for bucket in buckets:
    obj_list = s3.list_objects(Bucket=bucket)
    contents_list = obj_list['Contents']
    for content in contents_list:
        print(bucket, content['Key'])
