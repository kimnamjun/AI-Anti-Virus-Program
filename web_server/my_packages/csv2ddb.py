import csv
import boto3

dynamodb = boto3.resource('dynamodb')
tableName = 'AVA-01'  # FIXME
filename = './dataset/temp/pca_df.csv'  # FIXME

def csv_to_dynamo():
    csvfile = open(filename)

    write_to_dynamo(csv.DictReader(csvfile))

    return print("Done")

def write_to_dynamo(rows):
    table = dynamodb.Table(tableName)
    with table.batch_writer() as batch:
        for row in rows:
            batch.put_item(
                Item={
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
                }
            )
