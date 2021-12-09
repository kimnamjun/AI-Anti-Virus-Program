import boto3

s3 = boto3.client('s3')

buckets = ['ava-data-model', 'ava-data-json', 'ava-data-csv']
for bucket in buckets:
    obj_list = s3.list_objects(Bucket=bucket)
    contents_list = obj_list['Contents']
    for content in contents_list:
        print(bucket, content['Key'])
